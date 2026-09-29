"""
02_preprocess/lib/parser
─────────────────────────
Shared pcap parser layer.

  dpkt_parser  : one packet (raw incl. L2) → strip L2 + PktView from L3 (with span metadata)
                 + applies 4 masks (--ip_mask/--port_mask/--l3_mask/--l4_mask)
  pcap_source  : filelist-based session iteration (session/whole mode) + multiprocessing runner

--parser selection:
  Only the dpkt implementation is provided. The tshark parser only reserves the interface (NotImplementedError when parser_name="tshark").
"""
from . import dpkt_parser  # noqa: F401
from . import pcap_source  # noqa: F401
