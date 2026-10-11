| Run | Prompt | Notes found | Prefill tok/s (1st request) | TTFT 1st s | Decode tok/s (128-token summary) | Experts from VRAM (summary) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| q36-ctx48k | 8175 | 6/6 | 689 | 11.9 | 38.0 | 0.854 |
| q36-ctx48k | 32751 | 6/6 | 1192 | 27.5 | 47.0 | 0.939 |
| q36-ctx48k | 44983 | 6/6 | 1190 | 37.8 | 46.0 | 0.934 |
| q36-ctx48k: load 26.983500003814697 s, peak {'vram_dedicated_mib': 12985, 'vram_shared_mib': 1945, 'ram_private_mib': 11622, 'ram_ws_mib': 17148}, pagefile 2890 -> 2886 MiB |||||||
| q36-ctx128k | 65519 | 6/6 | 1037 | 63.2 | 24.8 | 0.76 |
| q36-ctx128k | 130883 | 6/6 | 768 | 170.5 | 24.2 | 0.837 |
| q36-ctx128k: load 24.91822838783264 s, peak {'vram_dedicated_mib': 12985, 'vram_shared_mib': 1785, 'ram_private_mib': 9958, 'ram_ws_mib': 18581}, pagefile 2886 -> 2658 MiB |||||||
| q36-ctx128k-stream | 65519 | 6/6 | 1005 | 65.2 | 16.9 | 0.855 |
| q36-ctx128k-stream | 130883 | 6/6 | 712 | 183.8 | 9.1 | 0.881 |
| q36-ctx128k-stream: load 28.732938051223755 s, peak {'vram_dedicated_mib': 12989, 'vram_shared_mib': 3547, 'ram_private_mib': 13283, 'ram_ws_mib': 18892}, pagefile 2658 -> 2660 MiB |||||||
| q36-ctx128k-stream-nodraft | 65519 | 6/6 | 994 | 66.0 | 15.8 | 0.864 |
| q36-ctx128k-stream-nodraft | 130883 | 6/6 | 713 | 183.6 | 8.0 | 0.908 |
| q36-ctx128k-stream-nodraft: load 35.45132279396057 s, peak {'vram_dedicated_mib': 12963, 'vram_shared_mib': 3387, 'ram_private_mib': 13274, 'ram_ws_mib': 18139}, pagefile 2991 -> 2878 MiB |||||||
| q35-ctx128k | 65519 | 6/6 | 1109 | 59.1 | 40.6 | 0.892 |
| q35-ctx128k | 130883 | 6/6 | 784 | 167.0 | 30.1 | 0.941 |
| q35-ctx128k: load 33.33908486366272 s, peak {'vram_dedicated_mib': 12985, 'vram_shared_mib': 1786, 'ram_private_mib': 10462, 'ram_ws_mib': 12122}, pagefile 2660 -> 2660 MiB |||||||
| q36-ctx262k | 260983 | 6/6 | 497 | 525.7 | 13.8 | 0.57 |
| q36-ctx262k: load 25.065102338790894 s, peak {'vram_dedicated_mib': 12985, 'vram_shared_mib': 1785, 'ram_private_mib': 12777, 'ram_ws_mib': 19293}, pagefile 2660 -> 2691 MiB |||||||
| q36-ctx262k-stream | 260983 | 6/6 | 460 | 566.8 | 3.7 | 0.847 |
| q36-ctx262k-stream: load 31.861448287963867 s, peak {'vram_dedicated_mib': 12995, 'vram_shared_mib': 6363, 'ram_private_mib': 15846, 'ram_ws_mib': 21262}, pagefile 2691 -> 3618 MiB |||||||
