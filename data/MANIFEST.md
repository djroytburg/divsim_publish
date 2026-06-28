# Data Manifest — divsim paper runs

Reference data for all valid dyadic, triadic, and four-way runs used in the paper.
DBs live under `$RUNS_DIR` (dyadic/triadic as `sweep_*.db`; four-way under `$RUNS_DIR/tetradic/`).
They are **not** included in this bundle due to size (~200 GB total).

**Validity criteria:**
- Dyadic: ≥ 18 distinct commenter agents, ≥ 200 total comments
- Triadic: ≥ 26 distinct commenter agents, ≥ 200 total comments
- Four-way: 40 agents (10 per model), ≥ 200 comments, all four of {gpt-oss, Qwen, Magistral, GLM-4} present

**Run counts (current):** 100 dyadic + 89 triadic (five-model pool) + 24 four-way = 213 valid runs.

**Engagement metric:** the variance decomposition and content->engagement analyses use **comments per post**
(CPP = in-degree / number of posts), which removes the posting-volume exposure confound that inflates raw
in-degree. Analysis scripts use whichever runs are present under `$RUNS_DIR`.

---

## Dyadic runs
**Total valid runs: 100**

| Filename | Pair | Seed | n_comments | n_agents | size_bytes | mtime |
|----------|------|------|-----------|---------|------------|-------|
| `sweep_gem_gem_glm_s137.db` | GLM-4 × Gemma | 137 | 545 | 20 | 59256832 | 1779256940 |
| `sweep_gem_gem_glm_s271.db` | GLM-4 × Gemma | 271 | 564 | 20 | 67678208 | 1779259415 |
| `sweep_gem_gem_glm_s314.db` | GLM-4 × Gemma | 314 | 556 | 20 | 62390272 | 1779262424 |
| `sweep_gem_gem_glm_s42.db` | GLM-4 × Gemma | 42 | 561 | 20 | 65548288 | 1779253891 |
| `sweep_gem_gem_glm_s999.db` | GLM-4 × Gemma | 999 | 561 | 20 | 57278464 | 1779262942 |
| `sweep_gem_gem_mag_s137.db` | Gemma × Mag | 137 | 563 | 20 | 66887680 | 1779237487 |
| `sweep_gem_gem_mag_s271.db` | Gemma × Mag | 271 | 573 | 20 | 69795840 | 1779237563 |
| `sweep_gem_gem_mag_s314.db` | Gemma × Mag | 314 | 573 | 20 | 72994816 | 1779240779 |
| `sweep_gem_gem_mag_s42.db` | Gemma × Mag | 42 | 202 | 20 | 20865024 | 1779233607 |
| `sweep_gem_gem_mag_s999.db` | Gemma × Mag | 999 | 568 | 20 | 70995968 | 1779241344 |
| `sweep_gem_gem_oss_s137.db` | Gemma × gpt-oss | 137 | 558 | 20 | 81575936 | 1779289803 |
| `sweep_gem_gem_oss_s271.db` | Gemma × gpt-oss | 271 | 560 | 20 | 87035904 | 1779292575 |
| `sweep_gem_gem_oss_s314.db` | Gemma × gpt-oss | 314 | 565 | 20 | 84631552 | 1779292556 |
| `sweep_gem_gem_oss_s42.db` | Gemma × gpt-oss | 42 | 561 | 20 | 91570176 | 1779289808 |
| `sweep_gem_gem_oss_s999.db` | Gemma × gpt-oss | 999 | 550 | 20 | 77602816 | 1779296057 |
| `sweep_gem_gem_qwe_s137.db` | Gemma × Qwen | 137 | 536 | 20 | 56975360 | 1779238651 |
| `sweep_gem_gem_qwe_s271.db` | Gemma × Qwen | 271 | 566 | 20 | 60739584 | 1779242852 |
| `sweep_gem_gem_qwe_s314.db` | Gemma × Qwen | 314 | 554 | 20 | 58511360 | 1779244264 |
| `sweep_gem_gem_qwe_s42.db` | Gemma × Qwen | 42 | 568 | 20 | 61599744 | 1779238533 |
| `sweep_gem_gem_qwe_s999.db` | Gemma × Qwen | 999 | 560 | 20 | 63430656 | 1779247673 |
| `sweep_gem_glm_gem_s137.db` | GLM-4 × Gemma | 137 | 554 | 20 | 71245824 | 1779265322 |
| `sweep_gem_glm_gem_s271.db` | GLM-4 × Gemma | 271 | 560 | 20 | 66015232 | 1779267644 |
| `sweep_gem_glm_gem_s314.db` | GLM-4 × Gemma | 314 | 569 | 20 | 65216512 | 1779269154 |
| `sweep_gem_glm_gem_s42.db` | GLM-4 × Gemma | 42 | 568 | 20 | 64397312 | 1779264496 |
| `sweep_gem_glm_gem_s999.db` | GLM-4 × Gemma | 999 | 570 | 20 | 70668288 | 1779270025 |
| `sweep_gem_mag_gem_s137.db` | Gemma × Mag | 137 | 576 | 20 | 67174400 | 1779244801 |
| `sweep_gem_mag_gem_s271.db` | Gemma × Mag | 271 | 557 | 20 | 64151552 | 1779248268 |
| `sweep_gem_mag_gem_s314.db` | Gemma × Mag | 314 | 569 | 20 | 70443008 | 1779248674 |
| `sweep_gem_mag_gem_s42.db` | Gemma × Mag | 42 | 576 | 20 | 71020544 | 1779244365 |
| `sweep_gem_mag_gem_s999.db` | Gemma × Mag | 999 | 571 | 20 | 68837376 | 1779252081 |
| `sweep_gem_oss_gem_s137.db` | Gemma × gpt-oss | 137 | 550 | 20 | 72888320 | 1779702718 |
| `sweep_gem_oss_gem_s271.db` | Gemma × gpt-oss | 271 | 538 | 20 | 75186176 | 1779703920 |
| `sweep_gem_oss_gem_s314.db` | Gemma × gpt-oss | 314 | 570 | 20 | 86704128 | 1779702754 |
| `sweep_gem_oss_gem_s42.db` | Gemma × gpt-oss | 42 | 564 | 20 | 92504064 | 1779296113 |
| `sweep_gem_oss_gem_s999.db` | Gemma × gpt-oss | 999 | 563 | 20 | 76296192 | 1779702748 |
| `sweep_gem_qwe_gem_s137.db` | Gemma × Qwen | 137 | 562 | 20 | 60616704 | 1779252242 |
| `sweep_gem_qwe_gem_s271.db` | Gemma × Qwen | 271 | 555 | 20 | 54685696 | 1779252843 |
| `sweep_gem_qwe_gem_s314.db` | Gemma × Qwen | 314 | 569 | 20 | 62332928 | 1779257273 |
| `sweep_gem_qwe_gem_s42.db` | Gemma × Qwen | 42 | 568 | 20 | 63574016 | 1779248639 |
| `sweep_gem_qwe_gem_s999.db` | Gemma × Qwen | 999 | 562 | 20 | 60620800 | 1779257045 |
| `sweep_glm_mag_AB_s137.db` | GLM-4 × Mag | 137 | 568 | 20 | 66908160 | 1779153376 |
| `sweep_glm_mag_AB_s271.db` | GLM-4 × Mag | 271 | 576 | 20 | 69083136 | 1779153711 |
| `sweep_glm_mag_AB_s314.db` | GLM-4 × Mag | 314 | 561 | 20 | 70963200 | 1779152748 |
| `sweep_glm_mag_AB_s42.db` | GLM-4 × Mag | 42 | 569 | 20 | 70332416 | 1779151835 |
| `sweep_glm_mag_AB_s999.db` | GLM-4 × Mag | 999 | 575 | 20 | 64864256 | 1779153345 |
| `sweep_glm_mag_BA_s137.db` | GLM-4 × Mag | 137 | 566 | 20 | 66088960 | 1779153390 |
| `sweep_glm_mag_BA_s271.db` | GLM-4 × Mag | 271 | 564 | 20 | 65060864 | 1779153084 |
| `sweep_glm_mag_BA_s314.db` | GLM-4 × Mag | 314 | 569 | 20 | 64892928 | 1779153443 |
| `sweep_glm_mag_BA_s42.db` | GLM-4 × Mag | 42 | 560 | 20 | 62205952 | 1779151252 |
| `sweep_glm_mag_BA_s999.db` | GLM-4 × Mag | 999 | 576 | 20 | 64098304 | 1779152861 |
| `sweep_glm_oss_AB_s137.db` | GLM-4 × gpt-oss | 137 | 538 | 20 | 68145152 | 1779125984 |
| `sweep_glm_oss_AB_s271.db` | GLM-4 × gpt-oss | 271 | 528 | 20 | 65155072 | 1779131809 |
| `sweep_glm_oss_AB_s314.db` | GLM-4 × gpt-oss | 314 | 499 | 20 | 58146816 | 1779132567 |
| `sweep_glm_oss_AB_s42.db` | GLM-4 × gpt-oss | 42 | 518 | 20 | 72134656 | 1779121310 |
| `sweep_glm_oss_AB_s999.db` | GLM-4 × gpt-oss | 999 | 532 | 20 | 60399616 | 1779141456 |
| `sweep_glm_oss_BA_s137.db` | GLM-4 × gpt-oss | 137 | 488 | 20 | 57929728 | 1779125732 |
| `sweep_glm_oss_BA_s271.db` | GLM-4 × gpt-oss | 271 | 542 | 20 | 71491584 | 1779130457 |
| `sweep_glm_oss_BA_s314.db` | GLM-4 × gpt-oss | 314 | 523 | 20 | 60743680 | 1779136288 |
| `sweep_glm_oss_BA_s42.db` | GLM-4 × gpt-oss | 42 | 521 | 20 | 65347584 | 1779120938 |
| `sweep_glm_oss_BA_s999.db` | GLM-4 × gpt-oss | 999 | 516 | 20 | 57565184 | 1779141325 |
| `sweep_glm_qwen_AB_s137.db` | GLM-4 × Qwen | 137 | 493 | 20 | 50180096 | 1779134514 |
| `sweep_glm_qwen_AB_s271.db` | GLM-4 × Qwen | 271 | 487 | 20 | 50466816 | 1779148018 |
| `sweep_glm_qwen_AB_s314.db` | GLM-4 × Qwen | 314 | 504 | 20 | 49885184 | 1779160411 |
| `sweep_glm_qwen_AB_s42.db` | GLM-4 × Qwen | 42 | 480 | 20 | 48644096 | 1779123126 |
| `sweep_glm_qwen_AB_s999.db` | GLM-4 × Qwen | 999 | 484 | 20 | 46813184 | 1779116576 |
| `sweep_glm_qwen_BA_s137.db` | GLM-4 × Qwen | 137 | 515 | 20 | 58732544 | 1779139938 |
| `sweep_glm_qwen_BA_s271.db` | GLM-4 × Qwen | 271 | 503 | 20 | 44761088 | 1779159052 |
| `sweep_glm_qwen_BA_s314.db` | GLM-4 × Qwen | 314 | 494 | 20 | 41005056 | 1779158811 |
| `sweep_glm_qwen_BA_s42.db` | GLM-4 × Qwen | 42 | 430 | 20 | 51204096 | 1779129622 |
| `sweep_glm_qwen_BA_s999.db` | GLM-4 × Qwen | 999 | 465 | 20 | 47808512 | 1779116069 |
| `sweep_gpt_mag_s137.db` | Mag × gpt-oss | 137 | 544 | 20 | 81920000 | 1779099209 |
| `sweep_gpt_mag_s271.db` | Mag × gpt-oss | 271 | 547 | 20 | 84930560 | 1779100844 |
| `sweep_gpt_mag_s314.db` | Mag × gpt-oss | 314 | 572 | 20 | 85561344 | 1779101218 |
| `sweep_gpt_mag_s42.db` | Mag × gpt-oss | 42 | 574 | 20 | 91918336 | 1779099281 |
| `sweep_gpt_mag_s999.db` | Mag × gpt-oss | 999 | 581 | 20 | 74018816 | 1779102413 |
| `sweep_gpt_qwe_s137.db` | Qwen × gpt-oss | 137 | 543 | 20 | 68038656 | 1779080909 |
| `sweep_gpt_qwe_s271.db` | Qwen × gpt-oss | 271 | 531 | 20 | 56549376 | 1779085792 |
| `sweep_gpt_qwe_s314.db` | Qwen × gpt-oss | 314 | 532 | 20 | 62742528 | 1779091564 |
| `sweep_gpt_qwe_s42.db` | Qwen × gpt-oss | 42 | 529 | 20 | 69840896 | 1779075601 |
| `sweep_gpt_qwe_s999.db` | Qwen × gpt-oss | 999 | 530 | 20 | 49586176 | 1779097052 |
| `sweep_mag_gpt_s137.db` | Mag × gpt-oss | 137 | 554 | 20 | 80723968 | 1779104432 |
| `sweep_mag_gpt_s271.db` | Mag × gpt-oss | 271 | 555 | 20 | 70488064 | 1779104670 |
| `sweep_mag_gpt_s314.db` | Mag × gpt-oss | 314 | 559 | 20 | 81272832 | 1779106080 |
| `sweep_mag_gpt_s42.db` | Mag × gpt-oss | 42 | 545 | 20 | 88379392 | 1779102897 |
| `sweep_mag_gpt_s999.db` | Mag × gpt-oss | 999 | 566 | 20 | 81620992 | 1779106413 |
| `sweep_mag_qwe_s137.db` | Mag × Qwen | 137 | 576 | 20 | 62140416 | 1779091491 |
| `sweep_mag_qwe_s271.db` | Mag × Qwen | 271 | 568 | 20 | 66150400 | 1779091847 |
| `sweep_mag_qwe_s314.db` | Mag × Qwen | 314 | 584 | 20 | 66252800 | 1779096942 |
| `sweep_mag_qwe_s42.db` | Mag × Qwen | 42 | 585 | 20 | 60993536 | 1779085988 |
| `sweep_mag_qwe_s999.db` | Mag × Qwen | 999 | 582 | 20 | 64733184 | 1779097088 |
| `sweep_qwe_gpt_s137.db` | Qwen × gpt-oss | 137 | 521 | 20 | 56729600 | 1779080901 |
| `sweep_qwe_gpt_s271.db` | Qwen × gpt-oss | 271 | 526 | 20 | 55017472 | 1779086001 |
| `sweep_qwe_gpt_s314.db` | Qwen × gpt-oss | 314 | 531 | 20 | 58732544 | 1779091698 |
| `sweep_qwe_gpt_s42.db` | Qwen × gpt-oss | 42 | 538 | 20 | 72736768 | 1779075498 |
| `sweep_qwe_gpt_s999.db` | Qwen × gpt-oss | 999 | 539 | 20 | 66531328 | 1779097026 |
| `sweep_qwe_mag_s137.db` | Mag × Qwen | 137 | 576 | 20 | 66981888 | 1779075592 |
| `sweep_qwe_mag_s271.db` | Mag × Qwen | 271 | 581 | 20 | 64507904 | 1779080944 |
| `sweep_qwe_mag_s314.db` | Mag × Qwen | 314 | 574 | 20 | 66420736 | 1779080997 |
| `sweep_qwe_mag_s42.db` | Mag × Qwen | 42 | 586 | 20 | 75513856 | 1779075559 |
| `sweep_qwe_mag_s999.db` | Mag × Qwen | 999 | 577 | 20 | 62619648 | 1779085857 |

## Triadic runs
**Total valid runs: 89**

| Filename | Triplet | Seed | n_comments | n_agents | size_bytes | mtime |
|----------|---------|------|-----------|---------|------------|-------|
| `sweep_tri_gem_mag_glm_s271.db` | GLM-4 + Gemma + Mag | 271 | 927 | 30 | 107360256 | 1779741352 |
| `sweep_tri_gem_mag_glm_s42.db` | GLM-4 + Gemma + Mag | 42 | 851 | 30 | 93626368 | 1779738907 |
| `sweep_tri_gem_mag_glm_s999.db` | GLM-4 + Gemma + Mag | 999 | 914 | 30 | 98635776 | 1779740176 |
| `sweep_tri_gem_oss_glm_s271.db` | GLM-4 + Gemma + gpt-oss | 271 | 853 | 30 | 112611328 | 1779679559 |
| `sweep_tri_gem_oss_glm_s42.db` | GLM-4 + Gemma + gpt-oss | 42 | 835 | 30 | 111042560 | 1779679853 |
| `sweep_tri_gem_oss_glm_s999.db` | GLM-4 + Gemma + gpt-oss | 999 | 836 | 30 | 104181760 | 1779681507 |
| `sweep_tri_gem_oss_mag_s271.db` | Gemma + Mag + gpt-oss | 271 | 860 | 30 | 116637696 | 1779740304 |
| `sweep_tri_gem_oss_mag_s42.db` | Gemma + Mag + gpt-oss | 42 | 857 | 30 | 118136832 | 1779735837 |
| `sweep_tri_gem_oss_mag_s999.db` | Gemma + Mag + gpt-oss | 999 | 865 | 30 | 113467392 | 1779740150 |
| `sweep_tri_gem_qwe_glm_s271.db` | GLM-4 + Gemma + Qwen | 271 | 851 | 30 | 88444928 | 1779663231 |
| `sweep_tri_gem_qwe_glm_s999.db` | GLM-4 + Gemma + Qwen | 999 | 829 | 30 | 87187456 | 1779662986 |
| `sweep_tri_gem_qwe_mag_s271.db` | Gemma + Mag + Qwen | 271 | 866 | 30 | 95408128 | 1779707852 |
| `sweep_tri_gem_qwe_mag_s42.db` | Gemma + Mag + Qwen | 42 | 846 | 30 | 84951040 | 1779705394 |
| `sweep_tri_gem_qwe_mag_s999.db` | Gemma + Mag + Qwen | 999 | 837 | 30 | 91226112 | 1779708400 |
| `sweep_tri_gem_qwe_oss_s271.db` | Gemma + Qwen + gpt-oss | 271 | 866 | 30 | 101412864 | 1779615407 |
| `sweep_tri_gem_qwe_oss_s42.db` | Gemma + Qwen + gpt-oss | 42 | 838 | 30 | 98525184 | 1779611501 |
| `sweep_tri_gem_qwe_oss_s999.db` | Gemma + Qwen + gpt-oss | 999 | 858 | 30 | 112046080 | 1779615839 |
| `sweep_tri_glm_mag_gem_s271.db` | GLM-4 + Gemma + Mag | 271 | 846 | 30 | 93204480 | 1779729956 |
| `sweep_tri_glm_mag_gem_s42.db` | GLM-4 + Gemma + Mag | 42 | 865 | 30 | 102367232 | 1779731904 |
| `sweep_tri_glm_mag_gem_s999.db` | GLM-4 + Gemma + Mag | 999 | 840 | 30 | 93835264 | 1779731115 |
| `sweep_tri_glm_oss_gem_s271.db` | GLM-4 + Gemma + gpt-oss | 271 | 862 | 30 | 108032000 | 1779672097 |
| `sweep_tri_glm_oss_gem_s42.db` | GLM-4 + Gemma + gpt-oss | 42 | 853 | 30 | 112455680 | 1779671509 |
| `sweep_tri_glm_oss_gem_s999.db` | GLM-4 + Gemma + gpt-oss | 999 | 850 | 30 | 113049600 | 1779674662 |
| `sweep_tri_glm_oss_mag_s271.db` | GLM-4 + Mag + gpt-oss | 271 | 836 | 30 | 96997376 | 1779583162 |
| `sweep_tri_glm_oss_mag_s42.db` | GLM-4 + Mag + gpt-oss | 42 | 850 | 30 | 105680896 | 1779576950 |
| `sweep_tri_glm_oss_mag_s999.db` | GLM-4 + Mag + gpt-oss | 999 | 842 | 30 | 97931264 | 1779584320 |
| `sweep_tri_glm_qwe_gem_s271.db` | GLM-4 + Gemma + Qwen | 271 | 670 | 30 | 67776512 | 1779633689 |
| `sweep_tri_glm_qwe_gem_s42.db` | GLM-4 + Gemma + Qwen | 42 | 813 | 30 | 76480512 | 1779631257 |
| `sweep_tri_glm_qwe_gem_s999.db` | GLM-4 + Gemma + Qwen | 999 | 799 | 30 | 85442560 | 1779633695 |
| `sweep_tri_glm_qwe_mag_s271.db` | GLM-4 + Mag + Qwen | 271 | 799 | 30 | 87195648 | 1779224711 |
| `sweep_tri_glm_qwe_mag_s42.db` | GLM-4 + Mag + Qwen | 42 | 833 | 30 | 106655744 | 1779219949 |
| `sweep_tri_glm_qwe_mag_s999.db` | GLM-4 + Mag + Qwen | 999 | 811 | 30 | 84738048 | 1779225552 |
| `sweep_tri_glm_qwe_oss_s271.db` | GLM-4 + Qwen + gpt-oss | 271 | 760 | 30 | 85078016 | 1779196637 |
| `sweep_tri_glm_qwe_oss_s42.db` | GLM-4 + Qwen + gpt-oss | 42 | 716 | 30 | 79732736 | 1779196192 |
| `sweep_tri_glm_qwe_oss_s999.db` | GLM-4 + Qwen + gpt-oss | 999 | 774 | 30 | 91947008 | 1779202782 |
| `sweep_tri_mag_glm_gem_s271.db` | GLM-4 + Gemma + Mag | 271 | 849 | 30 | 94908416 | 1779722016 |
| `sweep_tri_mag_glm_gem_s42.db` | GLM-4 + Gemma + Mag | 42 | 827 | 30 | 88150016 | 1779721920 |
| `sweep_tri_mag_glm_gem_s999.db` | GLM-4 + Gemma + Mag | 999 | 846 | 30 | 92332032 | 1779722279 |
| `sweep_tri_mag_oss_gem_s271.db` | Gemma + Mag + gpt-oss | 271 | 848 | 30 | 124346368 | 1779736648 |
| `sweep_tri_mag_oss_gem_s42.db` | Gemma + Mag + gpt-oss | 42 | 826 | 30 | 104566784 | 1779713437 |
| `sweep_tri_mag_oss_gem_s999.db` | Gemma + Mag + gpt-oss | 999 | 837 | 30 | 106917888 | 1779735944 |
| `sweep_tri_mag_oss_glm_s271.db` | GLM-4 + Mag + gpt-oss | 271 | 295 | 30 | 32628736 | 1779593348 |
| `sweep_tri_mag_oss_glm_s42.db` | GLM-4 + Mag + gpt-oss | 42 | 813 | 30 | 106270720 | 1779296502 |
| `sweep_tri_mag_oss_glm_s999.db` | GLM-4 + Mag + gpt-oss | 999 | 864 | 30 | 118476800 | 1779575944 |
| `sweep_tri_mag_qwe_gem_s271.db` | Gemma + Mag + Qwen | 271 | 868 | 30 | 102752256 | 1779700441 |
| `sweep_tri_mag_qwe_gem_s42.db` | Gemma + Mag + Qwen | 42 | 864 | 30 | 110419968 | 1779698791 |
| `sweep_tri_mag_qwe_gem_s999.db` | Gemma + Mag + Qwen | 999 | 866 | 30 | 104988672 | 1779701696 |
| `sweep_tri_mag_qwe_glm_s271.db` | GLM-4 + Mag + Qwen | 271 | 848 | 30 | 97263616 | 1779214062 |
| `sweep_tri_mag_qwe_glm_s42.db` | GLM-4 + Mag + Qwen | 42 | 822 | 30 | 91475968 | 1779214653 |
| `sweep_tri_mag_qwe_glm_s999.db` | GLM-4 + Mag + Qwen | 999 | 853 | 30 | 89014272 | 1779219381 |
| `sweep_tri_mag_qwe_oss_s271.db` | Mag + Qwen + gpt-oss | 271 | 794 | 30 | 100130816 | 1779170054 |
| `sweep_tri_mag_qwe_oss_s42.db` | Mag + Qwen + gpt-oss | 42 | 845 | 30 | 113418240 | 1779168353 |
| `sweep_tri_mag_qwe_oss_s999.db` | Mag + Qwen + gpt-oss | 999 | 824 | 30 | 94769152 | 1779172581 |
| `sweep_tri_oss_glm_gem_s271.db` | GLM-4 + Gemma + gpt-oss | 271 | 855 | 30 | 119349248 | 1779666150 |
| `sweep_tri_oss_glm_gem_s42.db` | GLM-4 + Gemma + gpt-oss | 42 | 834 | 30 | 102146048 | 1779664702 |
| `sweep_tri_oss_glm_gem_s999.db` | GLM-4 + Gemma + gpt-oss | 999 | 851 | 30 | 109056000 | 1779664188 |
| `sweep_tri_oss_mag_gem_s271.db` | Gemma + Mag + gpt-oss | 271 | 848 | 30 | 124026880 | 1779712359 |
| `sweep_tri_oss_mag_gem_s42.db` | Gemma + Mag + gpt-oss | 42 | 856 | 30 | 116924416 | 1779709320 |
| `sweep_tri_oss_mag_gem_s999.db` | Gemma + Mag + gpt-oss | 999 | 849 | 30 | 102916096 | 1779712561 |
| `sweep_tri_oss_mag_glm_s271.db` | GLM-4 + Mag + gpt-oss | 271 | 858 | 30 | 102461440 | 1779291369 |
| `sweep_tri_oss_mag_glm_s42.db` | GLM-4 + Mag + gpt-oss | 42 | 851 | 30 | 99733504 | 1779291731 |
| `sweep_tri_oss_mag_glm_s999.db` | GLM-4 + Mag + gpt-oss | 999 | 796 | 30 | 106704896 | 1779295866 |
| `sweep_tri_oss_qwe_gem_s271.db` | Gemma + Qwen + gpt-oss | 271 | 855 | 30 | 138854400 | 1779607702 |
| `sweep_tri_oss_qwe_gem_s42.db` | Gemma + Qwen + gpt-oss | 42 | 866 | 30 | 117444608 | 1779610048 |
| `sweep_tri_oss_qwe_gem_s999.db` | Gemma + Qwen + gpt-oss | 999 | 848 | 30 | 113299456 | 1779609179 |
| `sweep_tri_oss_qwe_glm_s271.db` | GLM-4 + Qwen + gpt-oss | 271 | 712 | 30 | 79577088 | 1779188985 |
| `sweep_tri_oss_qwe_glm_s42.db` | GLM-4 + Qwen + gpt-oss | 42 | 769 | 30 | 86376448 | 1779184057 |
| `sweep_tri_oss_qwe_glm_s999.db` | GLM-4 + Qwen + gpt-oss | 999 | 730 | 30 | 85274624 | 1779190791 |
| `sweep_tri_oss_qwe_mag_s271.db` | Mag + Qwen + gpt-oss | 271 | 854 | 30 | 109309952 | 1779164927 |
| `sweep_tri_oss_qwe_mag_s42.db` | Mag + Qwen + gpt-oss | 42 | 852 | 30 | 108691456 | 1779160752 |
| `sweep_tri_oss_qwe_mag_s999.db` | Mag + Qwen + gpt-oss | 999 | 859 | 30 | 102883328 | 1779166217 |
| `sweep_tri_qwe_glm_gem_s271.db` | GLM-4 + Gemma + Qwen | 271 | 829 | 30 | 94613504 | 1779625369 |
| `sweep_tri_qwe_glm_gem_s42.db` | GLM-4 + Gemma + Qwen | 42 | 818 | 30 | 73973760 | 1779621704 |
| `sweep_tri_qwe_glm_gem_s999.db` | GLM-4 + Gemma + Qwen | 999 | 830 | 30 | 81612800 | 1779625082 |
| `sweep_tri_qwe_mag_gem_s271.db` | Gemma + Mag + Qwen | 271 | 826 | 30 | 109219840 | 1779691521 |
| `sweep_tri_qwe_mag_gem_s42.db` | Gemma + Mag + Qwen | 42 | 845 | 30 | 100605952 | 1779692886 |
| `sweep_tri_qwe_mag_gem_s999.db` | Gemma + Mag + Qwen | 999 | 845 | 30 | 98328576 | 1779692494 |
| `sweep_tri_qwe_mag_glm_s271.db` | GLM-4 + Mag + Qwen | 271 | 834 | 30 | 81764352 | 1779208380 |
| `sweep_tri_qwe_mag_glm_s42.db` | GLM-4 + Mag + Qwen | 42 | 826 | 30 | 87134208 | 1779202593 |
| `sweep_tri_qwe_mag_glm_s999.db` | GLM-4 + Mag + Qwen | 999 | 825 | 30 | 89001984 | 1779208467 |
| `sweep_tri_qwe_oss_gem_s271.db` | Gemma + Qwen + gpt-oss | 271 | 842 | 30 | 102461440 | 1779598329 |
| `sweep_tri_qwe_oss_gem_s42.db` | Gemma + Qwen + gpt-oss | 42 | 861 | 30 | 116256768 | 1779599659 |
| `sweep_tri_qwe_oss_gem_s999.db` | Gemma + Qwen + gpt-oss | 999 | 844 | 30 | 107937792 | 1779610777 |
| `sweep_tri_qwe_oss_glm_s271.db` | GLM-4 + Qwen + gpt-oss | 271 | 767 | 30 | 88920064 | 1779177738 |
| `sweep_tri_qwe_oss_glm_s42.db` | GLM-4 + Qwen + gpt-oss | 42 | 736 | 30 | 82784256 | 1779176350 |
| `sweep_tri_qwe_oss_glm_s999.db` | GLM-4 + Qwen + gpt-oss | 999 | 765 | 30 | 98025472 | 1779181875 |
| `sweep_tri_qwe_oss_mag_s271.db` | Mag + Qwen + gpt-oss | 271 | 801 | 30 | 88059904 | 1779156383 |
| `sweep_tri_qwe_oss_mag_s42.db` | Mag + Qwen + gpt-oss | 42 | 863 | 30 | 127578112 | 1779156355 |
| `sweep_tri_qwe_oss_mag_s999.db` | Mag + Qwen + gpt-oss | 999 | 816 | 30 | 100204544 | 1779161905 |

---

## Four-way (tetradic) runs
**Total valid runs: 24** (2 seeds x 12 model->block rotations of {gpt-oss, Qwen, Magistral, GLM-4}; 40 agents, 10 per model). Stored under `$RUNS_DIR/tetradic/`.

| Filename | Seed | Rotation | n_comments | n_agents |
|----------|------|----------|-----------|---------|
| `tetra_s42_r1.db` | 42 | r1 | 1067 | 40 |
| `tetra_s42_r2.db` | 42 | r2 | 1088 | 40 |
| `tetra_s42_r3.db` | 42 | r3 | 1088 | 40 |
| `tetra_s42_r4.db` | 42 | r4 | 1090 | 40 |
| `tetra_s42_r5.db` | 42 | r5 | 1086 | 40 |
| `tetra_s42_r6.db` | 42 | r6 | 1023 | 40 |
| `tetra_s42_r7.db` | 42 | r7 | 1101 | 40 |
| `tetra_s42_r8.db` | 42 | r8 | 1016 | 40 |
| `tetra_s42_r9.db` | 42 | r9 | 1095 | 40 |
| `tetra_s42_r10.db` | 42 | r10 | 1080 | 40 |
| `tetra_s42_r11.db` | 42 | r11 | 1048 | 40 |
| `tetra_s42_r12.db` | 42 | r12 | 1076 | 40 |
| `tetra_s271_r1.db` | 271 | r1 | 1057 | 40 |
| `tetra_s271_r2.db` | 271 | r2 | 1029 | 40 |
| `tetra_s271_r3.db` | 271 | r3 | 1055 | 40 |
| `tetra_s271_r4.db` | 271 | r4 | 1249 | 40 |
| `tetra_s271_r5.db` | 271 | r5 | 1067 | 40 |
| `tetra_s271_r6.db` | 271 | r6 | 1082 | 40 |
| `tetra_s271_r7.db` | 271 | r7 | 1101 | 40 |
| `tetra_s271_r8.db` | 271 | r8 | 1102 | 40 |
| `tetra_s271_r9.db` | 271 | r9 | 1060 | 40 |
| `tetra_s271_r10.db` | 271 | r10 | 1079 | 40 |
| `tetra_s271_r11.db` | 271 | r11 | 1072 | 40 |
| `tetra_s271_r12.db` | 271 | r12 | 1113 | 40 |
