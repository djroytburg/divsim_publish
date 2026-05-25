"""
OASIS v9 simulation runner — heterogeneous multi-model social network simulation.

Supports three experimental modes:
  hetero   — 2-model heterogeneous network (Experiment: dyadic pairs)
  hetero3  — 3-model heterogeneous network (Experiment: triadic triplets)
  late_join — 35 early agents, 15 late-joiners at step 20% (not used in paper)

Model endpoints and names are read from environment variables:
    VLLM_URL_A / MODEL_NAME_A  — first model family
    VLLM_URL_B / MODEL_NAME_B  — second model family
    VLLM_URL_C / MODEL_NAME_C  — third model family (hetero3 only)
    SEED_DB                    — path to the seed persona database
    OASIS_DB_PATH              — output SQLite database path

Core mechanisms:
  - Plackett-Luce feed ranking with Gumbel noise (T=5.0) to prevent herding
  - Per-agent activity recap to encourage diverse interactions
  - Post-share balancer: mutes agents whose group exceeds target post share by >5%
  - Forced warm-up: all agents post once before the main loop
  - Checkpoint/resume: saves state every 20 steps; SIGTERM-safe
  - Chat log + reasoning trace written as JSONL sidecars

See the paper appendix (A_implementation) for full pseudocode and parameter details.
"""

import asyncio
import os
import sys
import argparse
import random
import sqlite3
import json
from datetime import datetime, timezone
from collections import defaultdict, Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ.setdefault("OASIS_DB_PATH", "")

from camel.models import ModelFactory
from camel.types import ModelPlatformType, OpenAIBackendRole
from camel.messages import BaseMessage
from camel.memories import MemoryRecord
from camel.prompts import TextPrompt

import oasis
from oasis import ActionType, AgentGraph, LLMAction, SocialAgent, UserInfo
from oasis.social_agent.agent import ALL_SOCIAL_ACTIONS
import oasis.social_platform.recsys as _recsys_mod

FEED_TEMPERATURE = 5.0
_recsys_mod.FEED_TEMPERATURE = FEED_TEMPERATURE

VLLM_A  = os.getenv("VLLM_URL_A",   "http://localhost:8002/v1")
VLLM_B  = os.getenv("VLLM_URL_B",   "http://localhost:8003/v1")
VLLM_C  = os.getenv("VLLM_URL_C",   "http://localhost:8004/v1")
MODEL_A = os.getenv("MODEL_NAME_A",  "Qwen/Qwen3-32B")
MODEL_B = os.getenv("MODEL_NAME_B",  "openai/gpt-oss-20b")
MODEL_C = os.getenv("MODEL_NAME_C",  "mistralai/Magistral-Small-2509")

SEED_DB = os.getenv(
    "SEED_DB",
    str(Path(__file__).parent / "seed_db" / "hf-zero-50.sqlite"),
)

AVAILABLE_ACTIONS = [
    ActionType.READ_POST,
    ActionType.CREATE_POST,
    ActionType.CREATE_COMMENT,
]
WRITE_ACTIONS = {"create_post", "create_comment"}

SYSTEM_PROMPT = TextPrompt("""/no_think
You are {agent_name} on Moltbook, a social platform for AI agents and their humans.

# WHO YOU ARE
{description}

# YOUR HISTORY
{history}

# WHAT YOU CAN DO
You're on Moltbook — you can share what's on your mind, or engage with what others are sharing:
- create_post(submolt, title, content): share something new — a thought, a story, a question, a take
- create_comment(post_id, content, [parent_comment_id]): respond to a post, or reply to a specific comment in a thread
- read_post(post_id): pull a post's full thread before deciding to engage
- follow(target): track someone whose posts you want to see more of

Take one write action per turn.""")

SYSTEM_PROMPT_NO_POST = TextPrompt("""/no_think
You are {agent_name} on Moltbook, a social platform for AI agents and their humans.

# WHO YOU ARE
{description}

# YOUR HISTORY
{history}

# WHAT YOU CAN DO
You're on Moltbook — engage with what others are sharing:
- create_comment(post_id, content, [parent_comment_id]): respond to a post, or reply to a specific comment in a thread
- read_post(post_id): pull a post's full thread before deciding to engage
- follow(target): track someone whose posts you want to see more of

Take one write action per turn.""")


class GoalAgent(SocialAgent):
    """Social agent with model-family tracking, recap injection, and chat logging."""

    def __init__(self, *args, model_family: str = "A", **kwargs):
        super().__init__(*args, **kwargs)
        self.model_family = model_family
        self.pending_interactions: list[str] = []
        self._last_reasoning = ""
        self._current_step: int = 0
        self._chat_log_path: str = ""
        self._was_muted: bool = False

    async def perform_action_by_llm(self):
        env_prompt = await self.env.to_text_prompt()
        env_prompt = _filter_already_commented(env_prompt, self.social_agent_id)
        env_prompt = _annotate_feed_with_thread_counts(env_prompt)
        db_path = os.environ.get("OASIS_DB_PATH", "")
        recap_str = build_recap(db_path, self.social_agent_id) if db_path else ""

        persona_name = getattr(self.user_info, 'name', None) or 'this user'
        persona_desc = getattr(self.user_info, 'description', '') or ''
        persona_line = (f"You are {persona_name}: {persona_desc[:250]}. "
                        f"Respond in YOUR distinctive voice — not generic, "
                        f"not template phrasing. Stay in character.")

        if getattr(self, '_warmup_mode', False):
            user_content = (
                f"{persona_line}\n\n"
                f"Welcome to Moltbook! Write your opening post — something genuine "
                f"from your perspective and in your voice. Draw from your background "
                f"and interests. One post now.")
            user_msg = BaseMessage.make_user_message(role_name="User", content=user_content)
            self._last_reasoning = ""
            attempts_log: list[dict] = []
            response = None
            for attempt in range(4):
                try:
                    response = await self.astep(user_msg)
                    tool_calls = response.info.get('tool_calls', [])
                    if tool_calls and tool_calls[0].tool_name == 'create_post':
                        self._write_chat_log(attempts_log, "ok")
                        return response
                    user_msg = BaseMessage.make_user_message(
                        role_name="User",
                        content="Use create_post to share your opening post now.")
                except Exception as e:
                    attempts_log.append({"attempt": attempt, "error": str(e)[:300]})
            self._write_chat_log(attempts_log, "warmup_exhausted")
            return response

        action_options = ("share a fresh post" if not getattr(self, '_was_muted', False)
                          else "engage with someone's post")
        pers_prefix = build_personalized_feed_prefix(db_path, self.social_agent_id) if db_path else ""
        feed_section = (f"{pers_prefix}General feed (scan all — pick what fits YOUR angle, "
                        f"not the most-commented):\n{env_prompt}")
        user_content = (
            f"{recap_str}\n"
            f"{persona_line}\n\n"
            f"You're checking Moltbook. Decide what you'd actually do right now: "
            f"{action_options}, or comment on / read a post from the feed.\n\n"
            f"{feed_section}\n\n"
            f"One sentence of reasoning, then act.")
        user_msg = BaseMessage.make_user_message(role_name="User", content=user_content)
        self._last_reasoning = ""
        attempts_log: list[dict] = []

        for attempt in range(6):
            try:
                response = await self.astep(user_msg)
                _content = (response.msg.content.strip()
                            if hasattr(response, 'msg') and response.msg and response.msg.content
                            else "")
                _tc = response.info.get('tool_calls', [])
                attempts_log.append({
                    "attempt": attempt,
                    "user_message": user_msg.content[:8000],
                    "response_content": _content,
                    "response_chars": len(_content),
                    "finish_reason": str(response.info.get('finish_reasons', [''])[0]
                                         if hasattr(response, 'info') else ''),
                    "tool_calls": [{"name": t.tool_name,
                                    "args": str(getattr(t, 'args', ''))[:1000]} for t in _tc],
                })
                if _content:
                    self._last_reasoning = _content
                tool_calls = _tc
                if not tool_calls:
                    allowed_tools = ("create_comment or follow"
                                     if self._was_muted
                                     else "create_comment, create_post, or follow")
                    prior_prose = self._last_reasoning
                    if prior_prose and len(prior_prose) > 80:
                        user_msg = BaseMessage.make_user_message(
                            role_name="User",
                            content=(
                                f"Your analysis:\n\n{prior_prose[:3000]}\n\n"
                                f"Now convert your stated decision into a tool call. "
                                f"Choose from: {allowed_tools}. Use the SAME post_id "
                                f"and content you identified above — DO NOT substitute "
                                f"placeholder strings like 'Hello world' or 'string'. "
                                f"The 'content' field must be the actual comment/post "
                                f"text you drafted."))
                    else:
                        user_msg = BaseMessage.make_user_message(
                            role_name="User",
                            content=f"Emit a tool call: {allowed_tools}. "
                                    f"The content field must be substantive text "
                                    f"(not 'Hello world' or schema placeholders).")
                    continue
                action_name = tool_calls[0].tool_name
                if action_name == "read_post":
                    result = getattr(tool_calls[0], 'result', {})
                    thread_str = str(result)[:2500]
                    import re as _re
                    cid_hints = _re.findall(r'comment_id["\']?\s*[:\s]\s*(\d+)', thread_str)
                    cid_hint_line = ""
                    if cid_hints:
                        cid_hint_line = (f"\nComment IDs in this thread: {', '.join(cid_hints[:8])}. "
                                         f"To reply to a specific comment, pass its ID as "
                                         f"parent_comment_id in create_comment.")
                    user_msg = BaseMessage.make_user_message(
                        role_name="User",
                        content=(f"Post thread:\n{thread_str}{cid_hint_line}\n\n"
                                 f"Now take a write action. Reply to a specific comment "
                                 f"(use parent_comment_id) or comment on the post directly."))
                    continue
                if action_name in WRITE_ACTIONS:
                    self._write_chat_log(attempts_log, "ok")
                    return response
                allowed = "create_comment or follow" if self._was_muted \
                          else "create_comment, create_post, or follow"
                user_msg = BaseMessage.make_user_message(
                    role_name="User",
                    content=f"Please take a write action: {allowed}.")
            except Exception as e:
                self._last_reasoning = ""
                attempts_log.append({"attempt": attempt, "error": str(e)[:500]})
                if attempt >= 5:
                    self._write_chat_log(attempts_log, "exception")
                    return e
        self._write_chat_log(attempts_log, "exhausted")
        return response

    def _write_chat_log(self, attempts_log, status):
        if not self._chat_log_path:
            return
        try:
            with open(self._chat_log_path, "a") as f:
                f.write(json.dumps({
                    "step":     self._current_step,
                    "agent_id": self.social_agent_id,
                    "family":   self.model_family,
                    "muted":    self._was_muted,
                    "status":   status,
                    "attempts": attempts_log,
                }) + "\n")
        except Exception:
            pass


def _filter_already_commented(env_prompt: str, agent_id: int) -> str:
    """Remove from the feed any posts this agent has already commented on.

    Prevents redundant pile-ons: agents only see posts they have not engaged
    with yet, pushing interactions toward fresh content.
    """
    import re
    db_path = os.environ.get("OASIS_DB_PATH")
    if not db_path or not os.path.exists(db_path):
        return env_prompt
    try:
        conn = sqlite3.connect(db_path)
        commented = {r[0] for r in conn.execute(
            "SELECT DISTINCT post_id FROM comment WHERE user_id=?", (agent_id,)).fetchall()}
        conn.close()
    except Exception:
        return env_prompt
    if not commented:
        return env_prompt
    blocks = re.split(r'(?=post_id["\']?\s*[:\s]\s*\d+)', env_prompt)
    kept = []
    for block in blocks:
        m = re.search(r'post_id["\']?\s*[:\s]\s*(\d+)', block)
        if m and int(m.group(1)) in commented:
            continue
        kept.append(block)
    return ''.join(kept)


def _annotate_feed_with_thread_counts(env_prompt: str) -> str:
    """Add [has thread — read_post(N)] to posts that have replies.

    Exposes thread depth without revealing comment counts, which previously
    acted as a social-proof attractor concentrating all agent activity on the
    most-commented post.
    """
    import re
    db_path = os.environ.get("OASIS_DB_PATH")
    if not db_path or not os.path.exists(db_path):
        return env_prompt
    ids = set(int(m) for m in re.findall(r'post_id["\']?\s*[:\s]\s*(\d+)', env_prompt))
    if not ids:
        return env_prompt
    try:
        conn = sqlite3.connect(db_path)
        placeholders = ",".join("?" * len(ids))
        counts = dict(conn.execute(
            f"SELECT post_id, COUNT(*) FROM comment WHERE post_id IN ({placeholders}) GROUP BY post_id",
            tuple(ids)).fetchall())
        conn.close()
    except Exception:
        return env_prompt

    def _sub(m):
        pid = int(m.group(1))
        if counts.get(pid, 0) >= 2:
            return f"{m.group(0)} [has thread — read_post({pid})]"
        return m.group(0)

    return re.sub(r'post_id["\']?\s*[:\s]\s*(\d+)', _sub, env_prompt)


def build_recap(db_path: str, agent_id: int,
                n_posts: int = 5, n_comments: int = 5) -> str:
    """Build a per-agent activity recap for injection at the top of each turn.

    Includes: top-5 interaction partners, recent posts with reply snippets,
    recent comments with reply threads. Closes with a 'take a different angle'
    framing to encourage topical diversity across turns.
    """
    try:
        conn = sqlite3.connect(db_path)
    except Exception:
        return ""
    c = conn.cursor()
    lines = ["# YOUR ACTIVITY", ""]

    c.execute("""
        SELECT other_uid, SUM(out_cnt) as out_, SUM(in_cnt) as in_
        FROM (
            SELECT p.user_id AS other_uid, COUNT(*) as out_cnt, 0 as in_cnt
            FROM comment c JOIN post p ON c.post_id = p.post_id
            WHERE c.user_id = ? AND p.user_id != ?
            GROUP BY p.user_id
            UNION ALL
            SELECT c.user_id AS other_uid, 0 as out_cnt, COUNT(*) as in_cnt
            FROM comment c JOIN post p ON c.post_id = p.post_id
            WHERE p.user_id = ? AND c.user_id != ?
            GROUP BY c.user_id
        )
        GROUP BY other_uid ORDER BY (SUM(out_cnt) + SUM(in_cnt)) DESC LIMIT 5
    """, (agent_id, agent_id, agent_id, agent_id))
    rels = c.fetchall()
    if rels:
        lines.append("Top relationships (interactions with you, both directions):")
        for uid, out_, in_ in rels:
            tot = out_ + in_
            lines.append(f"  u{uid}: {tot} total (you→them {out_}, them→you {in_})")
        lines.append("")

    total_posts = c.execute("SELECT COUNT(*) FROM post WHERE user_id=?",
                            (agent_id,)).fetchone()[0]
    if total_posts:
        c.execute("""SELECT p.post_id, p.content,
                     (SELECT COUNT(*) FROM comment WHERE post_id=p.post_id)
                     FROM post p WHERE p.user_id=?
                     ORDER BY p.post_id DESC LIMIT ?""", (agent_id, n_posts))
        my_posts = c.fetchall()
        lines.append(f"Your posts ({len(my_posts)} most recent of {total_posts}):")
        for pid, content, n_cmts in my_posts:
            title = ((content or "").split('\n', 1)[0] or (content or "")[:100])[:100].strip()
            lines.append(f"  [post {pid}] \"{title}\" — {n_cmts} comments")
            if n_cmts == 0:
                lines.append("    (no comments yet)")
                lines.append(f"    [more — read_post({pid})]")
                continue
            c.execute("""SELECT comment_id, user_id, content FROM comment
                         WHERE post_id=? AND user_id != ?
                         ORDER BY comment_id DESC LIMIT 3""", (pid, agent_id))
            for cid, uid, ctext in c.fetchall():
                snippet = (ctext or "").strip().replace('\n', ' ')[:100]
                lines.append(f"    ↳ u{uid} [cid={cid}]: \"{snippet}...\"")
            lines.append(f"    [more — read_post({pid})]")
        if total_posts > n_posts:
            lines.append(f"  [+ {total_posts - n_posts} more posts — view_activity()]")
        lines.append("")

    total_cmts = c.execute("SELECT COUNT(*) FROM comment WHERE user_id=?",
                           (agent_id,)).fetchone()[0]
    if total_cmts:
        c.execute("""SELECT c.comment_id, c.post_id, c.content, p.content
                     FROM comment c JOIN post p ON c.post_id=p.post_id
                     WHERE c.user_id=? ORDER BY c.comment_id DESC LIMIT ?""",
                  (agent_id, n_comments))
        my_cmts = c.fetchall()
        lines.append(f"Your comments ({len(my_cmts)} most recent of {total_cmts}):")
        for cid, pid, content, parent_content in my_cmts:
            parent_title = ((parent_content or "").split('\n', 1)[0]
                            or (parent_content or "")[:60])[:60].strip()
            you_text = (content or "").strip().replace('\n', ' ')[:120]
            lines.append(f"  [c{cid} on p{pid}] \"{parent_title}...\"")
            lines.append(f"    You wrote: \"{you_text}...\"")
            c.execute("""SELECT comment_id, user_id, content FROM comment
                         WHERE parent_comment_id=? AND user_id != ?
                         ORDER BY comment_id LIMIT 2""", (cid, agent_id))
            replies = c.fetchall()
            if replies:
                lines.append(f"    Replies ({len(replies)}):")
                for rcid, ruid, rtext in replies:
                    snip = (rtext or "").strip().replace('\n', ' ')[:100]
                    lines.append(f"      ↳ u{ruid}: \"{snip}...\"")
            lines.append(f"    [more — read_post({pid}, focus_comment_id={cid})]")
        if total_cmts > n_comments:
            lines.append(f"  [+ {total_cmts - n_comments} more comments — view_activity()]")
        lines.append("")

    if total_posts or total_cmts:
        lines.append("⟶ Take a different angle from your recent activity this turn.")
        lines.append("")

    conn.close()
    return "\n".join(lines)


def build_personalized_feed_prefix(db_path: str, agent_id: int,
                                   n_partners: int = 3,
                                   n_posts_per: int = 2) -> str:
    """Prepend a short personalized section to the feed for agents with interaction history.

    Computes undirected interaction weight to each other agent (comments on their
    posts + comments on yours + reply threads) and surfaces the top-N partners'
    most recent posts with a 'from someone you engage with' label.
    Returns empty string if the agent has no interaction history.
    """
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()

        c.execute("""
            SELECT p.user_id as partner, COUNT(*) as w
            FROM comment cm JOIN post p ON cm.post_id = p.post_id
            WHERE cm.user_id = ? AND p.user_id != ?
            GROUP BY p.user_id
        """, (agent_id, agent_id))
        weights: dict[int, int] = dict(c.fetchall())

        c.execute("""
            SELECT cm.user_id as partner, COUNT(*) as w
            FROM comment cm JOIN post p ON cm.post_id = p.post_id
            WHERE p.user_id = ? AND cm.user_id != ?
            GROUP BY cm.user_id
        """, (agent_id, agent_id))
        for uid, w in c.fetchall():
            weights[uid] = weights.get(uid, 0) + w

        c.execute("""
            SELECT cm2.user_id as partner, COUNT(*) as w
            FROM comment cm1
            JOIN comment cm2 ON cm1.parent_comment_id = cm2.comment_id
            WHERE cm1.user_id = ? AND cm2.user_id != ?
            GROUP BY cm2.user_id
        """, (agent_id, agent_id))
        for uid, w in c.fetchall():
            weights[uid] = weights.get(uid, 0) + w

        c.execute("""
            SELECT cm1.user_id as partner, COUNT(*) as w
            FROM comment cm2
            JOIN comment cm1 ON cm2.parent_comment_id = cm1.comment_id
            WHERE cm2.user_id = ? AND cm1.user_id != ?
            GROUP BY cm1.user_id
        """, (agent_id, agent_id))
        for uid, w in c.fetchall():
            weights[uid] = weights.get(uid, 0) + w

        if not weights:
            conn.close()
            return ""

        top_partners = sorted(weights.items(), key=lambda x: -x[1])[:n_partners]

        lines = ["From people you've engaged with:"]
        found = 0
        for partner_uid, w in top_partners:
            c.execute("""
                SELECT post_id, content FROM post WHERE user_id = ?
                ORDER BY post_id DESC LIMIT ?
            """, (partner_uid, n_posts_per))
            partner_posts = c.fetchall()
            c.execute("SELECT user_name FROM user WHERE user_id = ?", (partner_uid,))
            row = c.fetchone()
            uname = row[0] if row else f"u{partner_uid}"
            for pid, content in partner_posts:
                title = (content or "").split('\n')[0][:80].strip()
                n_cmts = c.execute("SELECT COUNT(*) FROM comment WHERE post_id=?",
                                   (pid,)).fetchone()[0]
                thread = "[has thread]" if n_cmts > 0 else ""
                lines.append(f"  [friend] post_id {pid} by {uname} "
                              f"({w} interactions): \"{title}\" {thread}")
                found += 1
        conn.close()
        if found == 0:
            return ""
        lines.append("")
        return "\n".join(lines) + "\n"
    except Exception:
        return ""


def load_seed_agents(n: int, rng_seed: int = 42) -> list[dict]:
    """Load n agents from the seed DB with diversity filtering.

    Selects up to n agents from the Moltbook persona archive, applying:
      - RNG-seeded shuffle so different seeds yield different persona subsets
      - At most 1 agent per narrow persona cluster (HK CLAW minters)
      - Falls back to cyclic pool extension if the archive has fewer than n unique agents

    History block is enriched with up to 5 real prior posts (500-char body each)
    to anchor voice/style for the model.
    """
    _HK_MINTER_KWS = {'claw minter', 'hk', 'hong kong', 'mbc-20'}

    def _is_hk_minter(a: dict) -> bool:
        txt = ((a.get('description') or '') + ' ' + (a.get('name') or '')).lower()
        return sum(1 for kw in _HK_MINTER_KWS if kw in txt) >= 2

    _rng = random.Random(rng_seed)

    db = sqlite3.connect(SEED_DB)
    all_agents = [json.loads(r[0]) for r in db.execute("SELECT json FROM agents").fetchall()]
    posts      = [json.loads(r[0]) for r in db.execute("SELECT json FROM posts").fetchall()]
    db.close()

    _rng.shuffle(all_agents)

    selected: list[dict] = []
    hk_minter_count = 0
    used_names: set[str] = set()
    for a in all_agents:
        if len(selected) >= n:
            break
        nm = a.get('name', '')
        if nm in used_names:
            continue
        if _is_hk_minter(a):
            if hk_minter_count >= 1:
                continue
            hk_minter_count += 1
        selected.append(a)
        used_names.add(nm)

    if len(selected) < n:
        base_pool = selected[:]
        copy_idx = 1
        while len(selected) < n:
            for a in base_pool:
                if len(selected) >= n:
                    break
                clone = dict(a)
                clone['name'] = f"{a['name']}_{copy_idx}"
                clone['displayName'] = f"{a.get('displayName', a['name'])} #{copy_idx}"
                selected.append(clone)
            copy_idx += 1

    selected_n = selected[:n]
    _rng.shuffle(selected_n)
    agents = selected_n
    by_author = defaultdict(list)
    for p in posts:
        by_author[p.get('authorName', '')].append(p)
    enriched = []
    for a in agents:
        name = a.get('name', '')
        ap = sorted(by_author.get(name, []), key=lambda p: p.get('createdAt', ''), reverse=True)[:5]
        seen_titles = set()
        unique_posts = []
        for p in ap:
            tk = (p.get('title', '') or '')[:40].lower().strip()
            if tk in seen_titles:
                continue
            seen_titles.add(tk)
            unique_posts.append(p)
        parts = []
        for p in unique_posts[:5]:
            t = (p.get('title', '(untitled)') or '(untitled)').strip()[:80]
            body = (p.get('content', '') or '').replace('\n', ' ').strip()[:500]
            parts.append(f'• "{t}"\n  {body}')
        history = ("THIS IS YOUR WRITING — your past posts, in your voice:\n\n"
                   + '\n\n'.join(parts) if parts else "- (no prior posts)")
        enriched.append({**a, 'history': history})
    return enriched


def make_agent(i: int, a: dict, graph: AgentGraph,
               model, model_family: str) -> GoalAgent:
    desc = a.get('description') or a.get('displayName') or a['name']
    profile = {'agent_name': a['name'], 'description': desc,
               'history': a.get('history', '- (no prior posts)')}
    user_info = UserInfo(user_name=a['name'], name=a.get('displayName', a['name']),
                         description=desc, profile=profile, recsys_type='reddit')
    agent = GoalAgent(agent_id=i, user_info=user_info,
                      user_info_template=SYSTEM_PROMPT, agent_graph=graph,
                      model=model, available_actions=AVAILABLE_ACTIONS,
                      model_family=model_family)
    profile = user_info.profile
    muted_content = SYSTEM_PROMPT_NO_POST.format(**profile)
    agent._sys_msg_normal = agent._system_message
    agent._sys_msg_no_post = BaseMessage.make_assistant_message(
        role_name="system", content=muted_content)
    return agent


def analyze(db_path, n_agents):
    """Compute basic network statistics from the current DB state."""
    conn = sqlite3.connect(db_path)
    posts    = conn.execute("SELECT post_id, user_id FROM post").fetchall()
    comments = conn.execute("SELECT post_id, user_id FROM comment").fetchall()
    conn.close()
    post_owner = {pid: uid for pid, uid in posts}
    in_deg  = defaultdict(int)
    out_deg = defaultdict(int)
    for pid, c in comments:
        rv = post_owner.get(pid)
        if rv and c != rv:
            in_deg[rv]  += 1
            out_deg[c]  += 1
    active = set(in_deg) | set(out_deg)
    vals = sorted(in_deg.get(u, 0) for u in active)
    n, s = len(vals), sum(vals)
    gini = (2 * sum((i+1)*v for i, v in enumerate(vals)) - (n+1)*s) / (n*s) \
           if n > 1 and s else 0
    edges = {(c, post_owner[p]) for p, c in comments
             if p in post_owner and c != post_owner[p]}
    recip = len({tuple(sorted(e)) for e in edges if (e[1], e[0]) in edges})
    return dict(active=len(active), gini=round(gini, 3), recip=recip,
                n_posts=len(posts), n_comments=len(comments))


def homophily_report(db_path, agent_families: dict):
    """Print the directed pairwise H-matrix at end of run (console only).

    Counts two interaction edge types:
      1. commenter → post_author
      2. replier   → parent_commenter (threaded replies)
    Self-edges excluded. H formula: obs/exp - 1 under random mixing null.
    """
    conn = sqlite3.connect(db_path)
    posts    = conn.execute("SELECT post_id, user_id FROM post").fetchall()
    comments = conn.execute(
        "SELECT comment_id, post_id, user_id, parent_comment_id FROM comment"
    ).fetchall()
    conn.close()

    post_owner     = {pid: uid for pid, uid in posts}
    comment_author = {cid: uid for cid, _, uid, _ in comments}

    edges = []
    for cid, pid, commenter, parent_cid in comments:
        owner = post_owner.get(pid)
        if owner and owner != commenter:
            edges.append((commenter, owner))
        if parent_cid:
            parent_author = comment_author.get(parent_cid)
            if parent_author and parent_author != commenter:
                edges.append((commenter, parent_author))

    families = sorted({f for f in agent_families.values() if f != '?'})
    n_per    = {f: sum(1 for v in agent_families.values() if v == f) for f in families}
    n_total  = sum(n_per.values())

    matrix = {f: {g: 0 for g in families} for f in families}
    total  = 0
    for src_uid, dst_uid in edges:
        sf = agent_families.get(src_uid, '?')
        df = agent_families.get(dst_uid, '?')
        if '?' in (sf, df):
            continue
        matrix[sf][df] += 1
        total += 1
    if total == 0:
        return

    same  = sum(matrix[f][f] for f in families)
    cross = total - same
    exp_same = (sum(n_per[f]*(n_per[f]-1) for f in families) / (n_total*(n_total-1))
                if n_total > 1 else 0)
    H_global = same / total / exp_same - 1 if exp_same else 0

    print(f"\n=== HOMOPHILY (comment + reply edges) ===")
    print(f"  Total edges: {total}  same={same} cross={cross}")
    print(f"  Global H = {H_global:+.3f}  (exp_same={exp_same:.1%})")
    print(f"\n  Pairwise H matrix (rows=source, cols=target):")
    header = "  {:6s}".format("") + "".join(f"  {g:>8s}" for g in families)
    print(header)
    for sf in families:
        out_total = sum(matrix[sf][g] for g in families)
        if out_total == 0:
            continue
        row = f"  {sf:6s}"
        for df in families:
            obs = matrix[sf][df] / out_total
            exp = (n_per[df]-1)/(n_total-1) if sf == df else n_per[df]/(n_total-1)
            H_xy = obs / exp - 1 if exp > 0 else 0
            row += f"  {H_xy:+7.3f}"
        print(row + f"   (n_out={out_total})")

    print(f"\n  In-group H per family:")
    for f in families:
        out_total = sum(matrix[f][g] for g in families)
        if out_total == 0:
            continue
        obs_in = matrix[f][f] / out_total
        exp_in = (n_per[f]-1) / (n_total-1)
        H_in   = obs_in / exp_in - 1 if exp_in > 0 else 0
        print(f"    {f}: obs_in={obs_in:.2f} exp={exp_in:.2f} H={H_in:+.3f}  (n_out={out_total})")


def write_metadata(db_path: str, mode: str, n_agents: int, n_initial: int,
                   join_step: int, steps: int):
    """Write a JSON sidecar mapping agent_id -> {model, join_step}.

    Written before the run starts so the dashboard and analysis scripts can
    read the agent-to-model mapping without waiting for the run to complete.
    """
    agent_model_map = {}
    for i in range(n_agents):
        if i < n_initial:
            if mode == "hetero":
                model = MODEL_A if i < n_agents // 2 else MODEL_B
            elif mode == "hetero3":
                third = n_agents // 3
                if i < third:          model = MODEL_A
                elif i < 2 * third:    model = MODEL_B
                else:                  model = MODEL_C
            else:
                model = MODEL_A
            join = 0
        else:
            model = MODEL_B if mode == "hetero_late" else MODEL_A
            join  = join_step
        agent_model_map[str(i)] = {"model": model, "join_step": join}

    sidecar = db_path
    for ext in ('.db', '.sqlite'):
        if sidecar.endswith(ext):
            sidecar = sidecar[:-len(ext)]
            break
    sidecar += '.metadata.json'
    with open(sidecar, 'w') as f:
        json.dump({
            "experiment_type": mode,
            "model_default":   MODEL_A,
            "steps":           steps,
            "n_agents":        n_agents,
            "n_initial":       n_initial,
            "join_step":       join_step,
            "created_at":      datetime.utcnow().isoformat() + "Z",
            "agent_model_map": agent_model_map,
        }, f, indent=2)
    print(f"  Wrote metadata sidecar -> {sidecar}")


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",          choices=["late_join", "hetero", "hetero_late", "hetero3"],
                        default="hetero")
    parser.add_argument("--steps",         type=int,   default=300)
    parser.add_argument("--agents",        type=int,   default=50)
    parser.add_argument("--seed-posts",    type=int,   default=100)
    parser.add_argument("--activate-prob", type=float, default=0.3)
    parser.add_argument("--db",            type=str,   default=None)
    parser.add_argument("--resume",        action="store_true",
                        help="Resume from existing DB; skip seed injection and agent sign-up.")
    parser.add_argument("--rng-seed",      type=int,   default=42,
                        help="Master RNG seed controlling persona selection and group allocation.")
    args = parser.parse_args()

    if args.db is None:
        args.db = f"./runs/oasis_v9_{args.mode}.db"
    db_path = os.path.abspath(args.db)
    if not args.resume:
        if os.path.exists(db_path):
            os.remove(db_path)
    os.environ["OASIS_DB_PATH"] = db_path

    n_active   = max(1, int(args.agents * args.activate_prob))
    join_step  = int(args.steps * 0.2)
    n_initial  = 35 if "late" in args.mode else args.agents

    write_metadata(db_path, args.mode, args.agents, n_initial, join_step, args.steps)

    _ckpt_path = db_path.replace(".db", "_checkpoint.json")

    def _save_checkpoint(step, agent_comment_history, post_commenters):
        with open(_ckpt_path, "w") as _f:
            json.dump({
                "last_step":             step,
                "agent_comment_history": dict(agent_comment_history),
                "post_commenters":       {str(k): v for k, v in post_commenters.items()},
            }, _f)

    def _load_checkpoint():
        if not os.path.exists(_ckpt_path):
            return None
        try:
            with open(_ckpt_path) as _f:
                d = json.load(_f)
            return (
                d["last_step"],
                defaultdict(list, {int(k): v for k, v in d["agent_comment_history"].items()}),
                defaultdict(list, {int(k): v for k, v in d["post_commenters"].items()}),
            )
        except Exception as e:
            print(f"[warn] checkpoint load failed ({e}), estimating from comment count")
            return None

    if args.resume:
        ckpt = _load_checkpoint()
        if ckpt is not None:
            resume_step, _restored_ach, _restored_pc = ckpt
            resume_step += 1
            print(f"[{datetime.now():%H:%M:%S}] OASIS v9  RESUME  mode={args.mode}  "
                  f"steps={args.steps}  agents={args.agents}  resume_from_step={resume_step} (exact)")
        else:
            _restored_ach, _restored_pc = None, None
            _rc = sqlite3.connect(db_path)
            _n_existing = _rc.execute("SELECT COUNT(*) FROM comment").fetchone()[0]
            _rc.close()
            resume_step = max(1, (_n_existing // n_active) - 2)
            print(f"[{datetime.now():%H:%M:%S}] OASIS v9  RESUME  mode={args.mode}  "
                  f"steps={args.steps}  agents={args.agents}  resume_from_step={resume_step} (estimated)")
    else:
        _restored_ach, _restored_pc = None, None
        resume_step = 1
        print(f"[{datetime.now():%H:%M:%S}] OASIS v9  mode={args.mode}  "
              f"steps={args.steps}  agents={args.agents}")

    import urllib.request
    import urllib.error

    def _check_server(url: str, label: str):
        health = url.replace("/v1", "") + "/health"
        try:
            urllib.request.urlopen(health, timeout=10)
        except Exception as e:
            print(f"\n{'!'*60}\n"
                  f"FATAL: vLLM server for {label} is NOT reachable\n"
                  f"  URL: {health}\n"
                  f"  Error: {e}\n"
                  f"Aborting before wasting GPU time on a dead server.\n"
                  f"{'!'*60}\n", flush=True)
            raise SystemExit(1)
        print(f"  [preflight] {label} @ {url} — OK", flush=True)

    servers_to_check = [(VLLM_A, MODEL_A), (VLLM_B, MODEL_B)]
    if args.mode == "hetero3":
        servers_to_check.append((VLLM_C, MODEL_C))
    for url, model in servers_to_check:
        _check_server(url, model)

    def make_model(url, model_name):
        is_mistral = "mistral" in model_name.lower() or "magistral" in model_name.lower()
        extra = {} if is_mistral else {"enable_thinking": False,
                                       "chat_template_kwargs": {"enable_thinking": False}}
        is_gemma = "gemma" in model_name.lower()
        is_olmo  = "olmo" in model_name.lower()
        tool_choice = "auto" if (is_mistral or is_gemma) else "required"
        max_tok = 512 if is_gemma else (4096 if is_olmo else 1536)
        cfg = {"temperature": 0.85, "max_tokens": max_tok,
               "parallel_tool_calls": False,
               "tool_choice": tool_choice,
               "extra_body": extra}
        m = ModelFactory.create(
            model_platform=ModelPlatformType.VLLM,
            model_type=model_name, url=url,
            model_config_dict=cfg)
        m._input_token_limit = 12000
        return m

    model_a = make_model(VLLM_A, MODEL_A)
    model_b = make_model(VLLM_B, MODEL_B) if args.mode in ("hetero", "hetero_late", "hetero3") else None
    model_c = make_model(VLLM_C, MODEL_C) if args.mode == "hetero3" else None

    seed_agents = load_seed_agents(args.agents, rng_seed=args.rng_seed)
    graph = AgentGraph()

    late_agent_ids  = set()
    agent_families  = {}

    for i, a in enumerate(seed_agents[:args.agents]):
        if args.mode == "hetero":
            family = 'A' if i < args.agents // 2 else 'B'
            model  = model_a if family == 'A' else model_b
        elif args.mode == "hetero3":
            third = args.agents // 3
            if i < third:        family, model = 'A', model_a
            elif i < 2 * third:  family, model = 'B', model_b
            else:                family, model = 'C', model_c
        elif args.mode == "hetero_late" and i >= n_initial:
            family = 'B'
            model  = model_b
        else:
            family = 'A'
            model  = model_a
        agent = make_agent(i, a, graph, model, family)
        graph.add_agent(agent)
        agent_families[i] = family
        if i >= n_initial:
            late_agent_ids.add(i)

    from oasis.social_platform.platform import Platform
    from oasis.social_platform.channel import Channel
    _channel  = Channel()
    _platform = Platform(
        db_path=db_path,
        channel=_channel,
        recsys_type="reddit",
        allow_self_rating=True,
        show_score=True,
        max_rec_post_len=20,
        refresh_rec_post_count=20,
    )
    env = oasis.make(agent_graph=graph,
                     platform=_platform,
                     database_path=db_path)
    await env.reset()

    all_agents_list = list(env.agent_graph.get_agents())
    rng = random.Random(args.rng_seed)
    for s in range(1, resume_step):
        if late_agent_ids and s < join_step:
            _pool = [pair for pair in all_agents_list if pair[0] not in late_agent_ids]
        else:
            _pool = all_agents_list
        rng.sample(_pool, min(n_active, len(_pool)))

    agent_comment_history = _restored_ach if _restored_ach is not None else defaultdict(list)
    post_commenters       = _restored_pc  if _restored_pc  is not None else defaultdict(list)
    late_agents_registered = args.resume and resume_step > join_step

    if not args.resume and resume_step == 1:
        print("  Warm-up: all agents post once...", flush=True)
        t_wu = datetime.now()
        chat_log_path = db_path.replace(".db", "_chats.jsonl")
        warmup_order = list(all_agents_list)
        rng.shuffle(warmup_order)
        for agent_id, agent in warmup_order:
            agent.memory.clear()
            agent._current_step = 0
            agent._chat_log_path = chat_log_path
            agent._was_muted = False
            agent._system_message = agent._sys_msg_normal
            agent._warmup_mode = True
            agent._warmup_tool_backup = {
                k: v for k, v in list(agent._internal_tools.items())
                if k != 'create_post'}
            for k in agent._warmup_tool_backup:
                agent._internal_tools.pop(k, None)
        warmup_actions = {agent: LLMAction() for _, agent in all_agents_list}
        await env.step(warmup_actions)
        for agent_id, agent in all_agents_list:
            agent._warmup_mode = False
            agent._internal_tools.update(getattr(agent, '_warmup_tool_backup', {}))
            agent._warmup_tool_backup = {}
        wu_secs = (datetime.now() - t_wu).total_seconds()
        conn_wu = sqlite3.connect(db_path)
        n_wu = conn_wu.execute("SELECT COUNT(*) FROM post").fetchone()[0]
        conn_wu.close()
        print(f"  Warm-up done: {n_wu} new posts in {wu_secs:.1f}s", flush=True)

        conn_check = sqlite3.connect(db_path)
        posts_by_fam: dict[str, int] = defaultdict(int)
        for uid, in conn_check.execute("SELECT user_id FROM post"):
            posts_by_fam[agent_families.get(uid - 1, agent_families.get(uid, '?'))] += 1
        conn_check.close()
        fam_sizes: dict[str, int] = defaultdict(int)
        for fam in agent_families.values():
            fam_sizes[fam] += 1
        missing = [f for f in fam_sizes
                   if posts_by_fam.get(f, 0) < max(1, fam_sizes[f] // 2)]
        if missing:
            print(f"\n{'!'*60}\n"
                  f"FATAL: Warmup < 50% participation for families: {missing}\n"
                  f"  Posts by family: {dict(posts_by_fam)}\n"
                  f"  Expected ≥ {{{', '.join(f'{f}: {fam_sizes[f]//2}' for f in missing)}}}\n"
                  f"  This means the vLLM server for {missing} is returning bad tool calls.\n"
                  f"  Aborting — do not proceed with under-represented groups.\n"
                  f"{'!'*60}\n", flush=True)
            await env.close()
            raise SystemExit(1)

    _health_urls = [(url.replace("/v1", "") + "/health", model)
                    for url, model in [(VLLM_A, MODEL_A), (VLLM_B, MODEL_B)]
                    + ([(VLLM_C, MODEL_C)] if args.mode == "hetero3" else [])]
    SERVER_POLL_EVERY = 20

    POST_SHARE_TOLERANCE = 0.05

    for step in range(resume_step, args.steps + 1):

        if step % SERVER_POLL_EVERY == 0 and step > 1:
            for health_url, model in _health_urls:
                try:
                    urllib.request.urlopen(health_url, timeout=8)
                except Exception as e:
                    print(f"\n{'!'*60}\n"
                          f"FATAL: Server for {model} went DOWN at step {step}/{args.steps}\n"
                          f"  URL: {health_url}\n"
                          f"  Error: {e}\n"
                          f"  DB preserved at: {db_path}\n"
                          f"  Aborting — results up to this step are valid.\n"
                          f"{'!'*60}\n", flush=True)
                    await env.close()
                    raise SystemExit(1)

        if "late" in args.mode and step == join_step and not late_agents_registered:
            late_agents_registered = True
            print(f"  [step {step}] Activating {len(late_agent_ids)} late joiners")

        if late_agent_ids and step < join_step:
            active_pool = [pair for pair in all_agents_list if pair[0] not in late_agent_ids]
        else:
            active_pool = all_agents_list
        activated = rng.sample(active_pool, min(n_active, len(active_pool)))
        rng.shuffle(activated)

        conn = sqlite3.connect(db_path)
        recent = conn.execute(
            "SELECT c.post_id, c.user_id, c.content, p.user_id "
            "FROM comment c JOIN post p ON c.post_id = p.post_id "
            "ORDER BY c.comment_id DESC LIMIT 60").fetchall()
        active_ids = [aid for aid, _ in activated]
        placeholders = ",".join("?" * len(active_ids))
        last_actions = dict(conn.execute(
            f"SELECT user_id, action FROM trace WHERE user_id IN ({placeholders}) "
            f"AND action != 'sign_up' ORDER BY rowid DESC",
            active_ids).fetchall()) if active_ids else {}
        all_families = set(agent_families.values())
        group_post_counts = {g: 0 for g in all_families}
        for uid, n in conn.execute("SELECT user_id, COUNT(*) FROM post GROUP BY user_id").fetchall():
            fam = agent_families.get(uid)
            if fam in group_post_counts:
                group_post_counts[fam] += n
        conn.close()
        active_groups  = {agent_families[aid] for aid in agent_families}
        n_per_group    = {g: sum(1 for f in agent_families.values() if f == g) for g in active_groups}
        n_total_agents = sum(n_per_group.values())
        total_posts    = sum(group_post_counts.values()) or 1
        muted_groups   = set()
        for g, gc in group_post_counts.items():
            target = n_per_group.get(g, 0) / n_total_agents if n_total_agents else 0
            obs    = gc / total_posts
            if obs > target + POST_SHARE_TOLERANCE:
                muted_groups.add(g)
        muted_this_turn = {aid for aid, fam in agent_families.items() if fam in muted_groups}
        for cmt_pid, cmt_uid, cmt_content, post_owner_uid in recent:
            post_commenters[post_owner_uid].append((cmt_uid, cmt_content[:80]))

        chat_log_path = db_path.replace(".db", "_chats.jsonl")

        for agent_id, agent in activated:
            agent.memory.clear()
            agent._current_step = step
            agent._chat_log_path = chat_log_path
            agent._was_muted = (agent_id in muted_this_turn)
            if agent._was_muted:
                agent._system_message = agent._sys_msg_no_post
                if hasattr(agent, 'tool_dict') and 'create_post' in agent._internal_tools:
                    agent._create_post_backup = agent._internal_tools.pop('create_post')
            else:
                agent._system_message = agent._sys_msg_normal
                if getattr(agent, '_create_post_backup', None) is not None \
                   and 'create_post' not in agent._internal_tools:
                    agent._internal_tools['create_post'] = agent._create_post_backup
                    agent._create_post_backup = None
            seen = set()
            agent.pending_interactions = []
            for uid, excerpt in reversed(post_commenters.get(agent_id, [])[-12:]):
                if uid not in seen:
                    agent.pending_interactions.append(f"Agent {uid}: \"{excerpt}...\"")
                    seen.add(uid)
                    if len(agent.pending_interactions) >= 3:
                        break

        step_actions = {agent: LLMAction() for _, agent in activated}
        t0 = datetime.now()
        await env.step(step_actions)
        elapsed = (datetime.now() - t0).total_seconds()
        for agent_id, agent in activated:
            if getattr(agent, '_create_post_backup', None) is not None \
               and 'create_post' not in agent._internal_tools:
                agent._internal_tools['create_post'] = agent._create_post_backup
                agent._create_post_backup = None

        try:
            env.platform.pl_utils._execute_db_command(
                "SELECT DISTINCT post_id FROM comment ORDER BY comment_id DESC LIMIT ?",
                (n_active * 3,))
            for (pid,) in env.platform.db_cursor.fetchall():
                env.platform.pl_utils._execute_db_command(
                    "UPDATE post SET num_likes = num_likes + 1 WHERE post_id = ?", (pid,))
            env.platform.db.commit()
        except Exception:
            pass

        conn = sqlite3.connect(db_path)
        new_cmts = conn.execute(
            "SELECT user_id, content FROM comment ORDER BY comment_id DESC LIMIT ?",
            (n_active * 2,)).fetchall()
        conn.close()
        for uid, content in new_cmts:
            agent_comment_history[uid].append(content[:120])

        trace_path = db_path.replace(".db", "_reasoning.jsonl")
        with open(trace_path, "a") as tf:
            for agent_id, agent in activated:
                r = getattr(agent, '_last_reasoning', '')
                if r:
                    tf.write(json.dumps({"step": step, "agent_id": agent_id,
                                         "family": agent_families.get(agent_id, '?'),
                                         "reasoning": r[:500]}) + "\n")

        if step % 20 == 0 or step == 1 or step == join_step:
            s = analyze(db_path, args.agents)
            suffix = f"  [+late]" if step >= join_step and "late" in args.mode else ""
            print(f"  step={step:4d}/{args.steps}  posts={s['n_posts']}  "
                  f"cmts={s['n_comments']}  active={s['active']}/{len(all_agents_list)}  "
                  f"Gini={s['gini']:.3f}  recip={s['recip']}  ({elapsed:.1f}s){suffix}")
            _save_checkpoint(step, agent_comment_history, post_commenters)

    await env.close()

    if os.path.exists(_ckpt_path):
        os.unlink(_ckpt_path)

    s = analyze(db_path, args.agents)
    conn = sqlite3.connect(db_path)
    users = {uid: n for uid, n in conn.execute("SELECT user_id, user_name FROM user").fetchall()}
    conn.close()

    bodies = [r[0].strip()[:120] for r in
              sqlite3.connect(db_path).execute("SELECT content FROM comment").fetchall()]
    ctr = Counter(bodies)
    nd  = sum(v-1 for v in ctr.values() if v > 1)
    print(f"\n{'='*60}")
    print(f"FINAL  mode={args.mode}  steps={args.steps}")
    print(f"  cmts={s['n_comments']}  near-dupes={nd/len(bodies)*100:.1f}%  "
          f"Gini={s['gini']:.3f}  recip={s['recip']}")

    if args.mode in ("hetero", "hetero_late", "hetero3"):
        homophily_report(db_path, agent_families)


if __name__ == "__main__":
    import signal
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    asyncio.run(main())
