# Proveniencia dos resultados full_v0

- git_sha: `0d0d918a14d209ac08a3ffa2737ae41171d75b1c`
- freeze_sha256: `ca360bf61af1ab3dc50fda20724aad642b7a70b017d6501362576e081b4b47d7`
- gerado em: 2026-11-16T20:51:20Z
- Os manifests das predicoes trazem `release: "dataset_100k_v0"` (item G15 em `docs/specs/dl-implementation.md` e ADR 0006 §17); este arquivo recalcula todos os SHA-256 como verificacao independente, em vez de confiar no campo `file_sha256` dos manifests.

| arquivo | bytes | sha256 | tempo (UTC) | fonte do tempo |
|---|---|---|---|---|
| `dl/eval/results_dl.md` | 12469 | `c86a887e7f8908b243979f594bab0685dc17b650aee3909f096afa4f5f5b9afe` | 2026-11-16T19:22:46Z | mtime |
| `dl/eval/results_dl.json` | 284341 | `037e684d2613142b1ed226489f492e2d4efb38a77fc9a06a4df777f5b1b0a3c6` | 2026-11-16T19:22:46Z | mtime |
| `dl/eval/risk_coverage_dl.csv` | 139932 | `42f3dea8f3e78fc293b5d40c957159c08d93b8338fa56c23770d20d2a395334a` | 2026-11-16T19:22:46Z | mtime |
| `dl/val_freeze.json` | 20190 | `ca360bf61af1ab3dc50fda20724aad642b7a70b017d6501362576e081b4b47d7` | 2026-11-16T19:10:02Z | mtime |
| `dl/test_opened.json` | 269 | `fe77a2e9b2961a0d022c5e32e7bb741077729663a81b636a967f26414de5b175` | 2026-11-16T19:22:46Z | opened_at |
| `dl/train_report_dl.json` | 72300 | `1d1d6efa51566971ee6ed2edf2257f456d2909ecfa434bce03cfeda1a0b30c38` | 2026-11-16T19:09:18Z | mtime |
| `dl/environment.json` | 136 | `465d348f3c9fb7b8ff64666ec6dad8fab35c3c1f18e59fa926c16be1643c3428` | 2026-11-09T15:21:55Z | mtime |
| `lgbm_seeds/delta.json` | 41244 | `a2b1785a9c41e74b0b37953c589cc5a1a7a8e8c884c164e0b6bfb8b8b609d5bb` | 2026-11-04T15:13:34Z | mtime |
| `lgbm_seeds/train_report_lgbm_seeds.json` | 14963 | `b66a203cf27697af4e87cdcac9c0da4dc7e3af9ce5b995a9bfa656697a7fe256` | 2026-11-04T14:58:35Z | mtime |
| `dl/predictions_ens_ftt_full_s0_v0.parquet` | 20386243 | `1935bef2380b813ce1bc78541b42a13d7274e027b1132235584b4e601bb0adfd` | 2026-11-16T19:08:59Z | obtained_utc |
| `dl/predictions_ens_ftt_full_s1_v0.parquet` | 20408347 | `c82aceb3df7fd1a0612f706f77fbdb72d5faab4a7d99f54a101f6a0cae2e4f36` | 2026-11-16T19:09:02Z | obtained_utc |
| `dl/predictions_ens_ftt_full_s2_v0.parquet` | 20413074 | `0e1485b0b1d8390bf045c853fe1be6f72ad171b3b1264c8f67cd5e4fe2a9d992` | 2026-11-16T19:09:07Z | obtained_utc |
| `dl/predictions_ens_ftt_full_s3_v0.parquet` | 20445222 | `875b5e2b0bfbe4bbbd474ae227c97f0e1cce3d263768e83122b286164baa678b` | 2026-11-16T19:09:11Z | obtained_utc |
| `dl/predictions_ens_ftt_full_s4_v0.parquet` | 20408482 | `1dbd30c304d06ec1431bda6e680fcd007537c5cf4b22f84f47e95f31d572485e` | 2026-11-16T19:09:17Z | obtained_utc |
| `dl/predictions_ens_mlp_full_s0_v0.parquet` | 20459704 | `c6d98fae6031f618b322c50248c76e47edc0f682bfc81bf6c6e85c75f0a97ae7` | 2026-11-16T19:08:41Z | obtained_utc |
| `dl/predictions_ens_mlp_full_s1_v0.parquet` | 20481326 | `669d142812d58bc44cb45d9769a22a827decc56efc3a20b411e8c731060fd82c` | 2026-11-16T19:08:45Z | obtained_utc |
| `dl/predictions_ens_mlp_full_s2_v0.parquet` | 20466981 | `f424be5a1e00b51d10fe433877b2415ac3925938bb80dde31ab29d9b21967bff` | 2026-11-16T19:08:48Z | obtained_utc |
| `dl/predictions_ens_mlp_full_s3_v0.parquet` | 20461005 | `f8c39718dbdd5cc4b45a5940b082d093f2dfb1badc55481775831c8fdb98c163` | 2026-11-16T19:08:52Z | obtained_utc |
| `dl/predictions_ens_mlp_full_s4_v0.parquet` | 20494194 | `13deb5bba4c7f0eef437a5393a43bf630981d3d29b658244168d7c20548aa928` | 2026-11-16T19:08:55Z | obtained_utc |
| `dl/predictions_ftt_full_s0_v0.parquet` | 20347235 | `bd85f057656f15607791c4ef987ad59265b4a3147981b02b99b8226f76450f2d` | 2026-11-09T21:35:35Z | obtained_utc |
| `dl/predictions_ftt_full_s1_v0.parquet` | 20376964 | `79287abc20ade0bd7a20fd49b301e7dc830eb408b93de7bb72d6c2e12586d698` | 2026-11-10T05:22:18Z | obtained_utc |
| `dl/predictions_ftt_full_s2_v0.parquet` | 20395733 | `07aeeae2089acb1e662662bf8881633a4e1f20fc44a7df4ec11ae114a12cea2c` | 2026-11-12T03:58:11Z | obtained_utc |
| `dl/predictions_ftt_full_s3_v0.parquet` | 20415983 | `13944f97f466e449beb62096b80c68f8df22b051bb7016bb579c50a58ad87c82` | 2026-11-12T15:45:22Z | obtained_utc |
| `dl/predictions_ftt_full_s4_v0.parquet` | 20381539 | `58b33f4043f1402051ad53d37d4f494fc45203aae5d8c9ea77ad72b4c596a89f` | 2026-11-16T13:38:26Z | obtained_utc |
| `dl/predictions_ftt_noind_s0_v0.parquet` | 20558052 | `6965e86c038f522e6051ba246a36c65d9c6f8a9bec8ddbd82a75d0187b07323e` | 2026-11-09T23:56:32Z | obtained_utc |
| `dl/predictions_ftt_noind_s1_v0.parquet` | 20556480 | `4ac237163c9d0548a46bd627d367653f296f2c21714de20c2127df8e5527e9ca` | 2026-11-10T07:55:34Z | obtained_utc |
| `dl/predictions_ftt_noind_s2_v0.parquet` | 20568986 | `b6c9160c6babc0688e12be28d0d06fca7ce70ed8a847b5423742284ffec38a10` | 2026-11-12T06:38:24Z | obtained_utc |
| `dl/predictions_ftt_noind_s3_v0.parquet` | 20571048 | `90ba55afb93b3c3ddefade80084ed2b2f1679b6ea22412986387f0e8677c1f49` | 2026-11-12T18:20:37Z | obtained_utc |
| `dl/predictions_ftt_noind_s4_v0.parquet` | 20537430 | `8285c9d9e96806d226533e3d7b1b3a74b3f41cc634f4fbdae07f67c5ac4d3663` | 2026-11-16T15:38:42Z | obtained_utc |
| `dl/predictions_mlp_full_s0_v0.parquet` | 20492429 | `b4ef9324dad01a0ca784752e5dfcda48e22f8dd819d6a91ef38a66d8a8cc2617` | 2026-11-09T21:38:33Z | obtained_utc |
| `dl/predictions_mlp_full_s1_v0.parquet` | 20527264 | `1bea0878589ff7c9767c63c055a6441e657d1cd4422b84f5c320338eb9840784` | 2026-11-10T05:26:01Z | obtained_utc |
| `dl/predictions_mlp_full_s2_v0.parquet` | 20527406 | `4e8b294ef6fd678ba9048cf48eac55253532f82b36841efe37807b71a85538cf` | 2026-11-12T04:01:00Z | obtained_utc |
| `dl/predictions_mlp_full_s3_v0.parquet` | 20512997 | `2403456919ae9d5c56405e11dbf4bf681686cbafbc6771ec29748d65401fc615` | 2026-11-12T15:48:37Z | obtained_utc |
| `dl/predictions_mlp_full_s4_v0.parquet` | 20544735 | `1df326772722ebf8f1c7fdba274aea6c81795a279492ec0e871ee9b9e55817b0` | 2026-11-16T13:41:36Z | obtained_utc |
| `dl/predictions_mlp_noind_s0_v0.parquet` | 20661618 | `a19c36ea42d6790b836625cab06729bba32dbb72c4db9b3551ebe6bf0128729f` | 2026-11-10T00:00:21Z | obtained_utc |
| `dl/predictions_mlp_noind_s1_v0.parquet` | 20670425 | `049cf4b325d27e58bd9aae4b7ae410a8c612c3b6b5207d120dbe6347da217f1b` | 2026-11-10T07:59:29Z | obtained_utc |
| `dl/predictions_mlp_noind_s2_v0.parquet` | 20686119 | `f300e4cfc4e3eeacb9cfa826e563b5292ba1b0945f68e1620b28feb51c40767b` | 2026-11-12T06:41:35Z | obtained_utc |
| `dl/predictions_mlp_noind_s3_v0.parquet` | 20666168 | `281ec7d4ee446b9dc045b0e0657ed484e53daaab8f1f0c23fea776b21cca7d23` | 2026-11-12T18:24:07Z | obtained_utc |
| `dl/predictions_mlp_noind_s4_v0.parquet` | 20652146 | `1326bcf404c4edf000bbc72325ee6f3ba3919ea0dfe53d52c9b3779f5cf88a13` | 2026-11-16T15:41:59Z | obtained_utc |
| `lgbm_seeds/predictions_lgbm_full_s0_v0.parquet` | 20456204 | `72478719648b516a2d96c7629e45daab404914470d7cb889b12f7bf4c0033b12` | 2026-11-03T20:37:12Z | obtained_utc |
| `lgbm_seeds/predictions_lgbm_full_s1_v0.parquet` | 20451425 | `fcf5d86c76dc1a53391b83cd5703c1e284e94fd836cfd56016a866b3eed70b98` | 2026-11-03T20:48:23Z | obtained_utc |
| `lgbm_seeds/predictions_lgbm_full_s2_v0.parquet` | 20454172 | `209962af3982a643f7d56c72d780ea1d8a7db8b3742c6da0ea441c333ec37cc3` | 2026-11-03T20:59:42Z | obtained_utc |
| `lgbm_seeds/predictions_lgbm_full_s3_v0.parquet` | 20452781 | `0d51fbd287c123df6939772c586c636197263c140ba9211fc6f7d35b7573f62e` | 2026-11-03T21:10:27Z | obtained_utc |
| `lgbm_seeds/predictions_lgbm_full_s4_v0.parquet` | 20456908 | `3a3724f7518374169c5511415460598916ce576e5cf255f3608bd3781af9410a` | 2026-11-03T21:22:13Z | obtained_utc |
| `lgbm_seeds/predictions_lgbm_noind_s0_v0.parquet` | 20592177 | `3df6e03b7d1af6ccd316cc98069a38b4a2da9973cf8043dff29fb4c460fbf24f` | 2026-11-03T21:33:43Z | obtained_utc |
| `lgbm_seeds/predictions_lgbm_noind_s1_v0.parquet` | 20601553 | `2c02d6e7244d08b7351eca5f3800b9aa9ed33225facbfa43e499ebfce1748d26` | 2026-11-03T21:44:31Z | obtained_utc |
| `lgbm_seeds/predictions_lgbm_noind_s2_v0.parquet` | 20597775 | `3c68a72b9912f6736cf17f141e93ae43379edc685074cabbbcaf61ca18aa2b0a` | 2026-11-03T21:54:56Z | obtained_utc |
| `lgbm_seeds/predictions_lgbm_noind_s3_v0.parquet` | 20602183 | `cf262bd25f8eeae967ce2d4f9fb1de1342569d1939481c0de6a17f6edeba1f9d` | 2026-11-03T22:06:13Z | obtained_utc |
| `lgbm_seeds/predictions_lgbm_noind_s4_v0.parquet` | 20601862 | `026bc48185957bf92073894ff798457995ca4a27219af018bb33893d61a216fc` | 2026-11-03T22:17:37Z | obtained_utc |
