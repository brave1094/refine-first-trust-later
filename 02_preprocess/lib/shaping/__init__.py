"""
02_preprocess/lib/shaping
─────────────────────────
Registry of per-model input transformation (shaping) modules.

Each shaping_{model}.py implements the following contract:

  make_sample(packets, meta, opt) -> sample | list[sample] | None
      packets : [(ts, buf, linktype), ...]  (passed by pcap_source, raw including L2)
      meta    : filelist row dict
      opt     : {"ip_mask","port_mask","l3_mask","l4_mask","parser","split_role",...}
      Runs in a worker process, so it must be a pure function.

  Writer(out_dir, opt)
      .add(meta, sample, label_idx)
      .close() -> n_written
      Runs in the main process (responsible for writing files).

Masking convention for byte-based models:
  dpkt_parser.masked_ints marks masked positions with -1 →
  each module replaces them with its own MASK value (2dcnn=257, other byte models=0x00).

xgboost / rf do not go through shaping and use the 1,205-feature extraction path
of 03_model/own_models/02_xgboost/extractor.py as is (see 06_make_dataset.py).
"""
BYTE_MODELS = ["2dcnn", "etbert", "yatc", "netmamba", "trafficformer", "mm4flow"]
FEAT_MODELS = ["xgboost", "rf"]
# NetFound: separate pcap→arrow (HF tokenizer) pipeline. Instead of the make_sample contract,
#   06 build_netfound arranges filelist sessions into {int_label}/raw/ → container preprocess → arrow.
NETFOUND_MODELS = ["netfound"]
ALL_MODELS = FEAT_MODELS + BYTE_MODELS + NETFOUND_MODELS
