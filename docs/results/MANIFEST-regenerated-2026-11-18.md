# Proveniencia dos resultados full_v0

- git_sha: `5201e401921f6661cbdf0f99a9bd056a619e5118`
- freeze_sha256: `a9d74658d6a6f1f0902186c8402f8263139d8d795847b791e13941c941ee8d47`
- gerado em: 2026-11-18T20:08:16Z
- Os manifests das predicoes trazem `release: "dataset_100k_v0"` (item G15 em `docs/specs/dl-implementation.md` e ADR 0006 §17); este arquivo recalcula todos os SHA-256 como verificacao independente, em vez de confiar no campo `file_sha256` dos manifests.

| arquivo | bytes | sha256 | tempo (UTC) | fonte do tempo |
|---|---|---|---|---|
| `dl/eval/results_dl.md` | 12600 | `68eb98b68c493796a7be641e774383c57a976059339ec0c2382e75e2e72c3b03` | 2026-11-18T19:57:32Z | mtime |
| `dl/eval/results_dl.json` | 284472 | `6e3cf051e9e9cddd3fe7c9f83476ca7fddf18bdb0e808eeac7660ed048288930` | 2026-11-18T19:57:32Z | mtime |
| `dl/eval/risk_coverage_dl.csv` | 139932 | `42f3dea8f3e78fc293b5d40c957159c08d93b8338fa56c23770d20d2a395334a` | 2026-11-18T19:57:32Z | mtime |
| `dl/val_freeze.json` | 20299 | `a9d74658d6a6f1f0902186c8402f8263139d8d795847b791e13941c941ee8d47` | 2026-11-18T19:46:11Z | mtime |
| `dl/test_opened.json` | 269 | `0c8d1cd62631565010c9b84318a61a3a6ab9e6e7748b734b24e628fd71d02462` | 2026-11-18T19:57:32Z | opened_at |
| `dl/train_report_dl.json` | 72300 | `5d809b77ee645452de9fa838fc8a8ca6e60d11dab2ee091ab4a32f9e6f92cf76` | 2026-11-17T20:17:13Z | mtime |
| `dl/environment.json` | 136 | `057223aee584ca92d7ea0f901dcdc463a377e3c924169207a9c53deca14ac6c0` | 2026-11-17T20:16:32Z | mtime |
| `lgbm_seeds/delta.json` | 41244 | `a2b1785a9c41e74b0b37953c589cc5a1a7a8e8c884c164e0b6bfb8b8b609d5bb` | 2026-11-17T19:58:49Z | mtime |
| `lgbm_seeds/train_report_lgbm_seeds.json` | 14963 | `b66a203cf27697af4e87cdcac9c0da4dc7e3af9ce5b995a9bfa656697a7fe256` | 2026-11-17T19:58:20Z | mtime |
| `dl/predictions_ens_ftt_full_s0_v0.parquet` | 20386243 | `1d1ba7df3e4e4de2a45bcc5c3981b30c60fb99000b780ac3084c97c90cb0ae82` | 2026-11-17T20:16:56Z | obtained_utc |
| `dl/predictions_ens_ftt_full_s1_v0.parquet` | 20408347 | `056840f9dd8154496e6ec11a30655e1ecf19fca8746c08f736c54cb452ad1ee7` | 2026-11-17T20:17:00Z | obtained_utc |
| `dl/predictions_ens_ftt_full_s2_v0.parquet` | 20413074 | `629cf2acf672885e59803c8b5eb2b781a17b3d067dfdbaeb4ae5d03ef923b0c6` | 2026-11-17T20:17:04Z | obtained_utc |
| `dl/predictions_ens_ftt_full_s3_v0.parquet` | 20445222 | `fbc2efff4f4bd94e10f2223c54427a4f8f6f927fe1de25bceff80f4f5d38e405` | 2026-11-17T20:17:08Z | obtained_utc |
| `dl/predictions_ens_ftt_full_s4_v0.parquet` | 20408482 | `35f92b26d79cde62781ebcbcd944672331262ead96e803f911b7ba9f66a99508` | 2026-11-17T20:17:12Z | obtained_utc |
| `dl/predictions_ens_mlp_full_s0_v0.parquet` | 20459704 | `af7ce92512114918cfeca1426128c41ce59aca1a9a193258df2efde1c17bd7e6` | 2026-11-17T20:16:36Z | obtained_utc |
| `dl/predictions_ens_mlp_full_s1_v0.parquet` | 20481326 | `6901389489d1d0020c48e8b6bc0e15e54530a074486bd0ed5debed18dbdeee67` | 2026-11-17T20:16:40Z | obtained_utc |
| `dl/predictions_ens_mlp_full_s2_v0.parquet` | 20466981 | `b7f9a2e111e841ffae7800a531c29b391ebf04fc4e0d3b5cda34a0796ba3f497` | 2026-11-17T20:16:44Z | obtained_utc |
| `dl/predictions_ens_mlp_full_s3_v0.parquet` | 20461005 | `3a8d13570b070c41ab79da92da40fad144bc005ef09cca7309acb73e15db30ef` | 2026-11-17T20:16:48Z | obtained_utc |
| `dl/predictions_ens_mlp_full_s4_v0.parquet` | 20494194 | `12fbd4957a258f53b75a7319a51f308dbdfe41215757b59ef52cf2821ea985cf` | 2026-11-17T20:16:53Z | obtained_utc |
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
