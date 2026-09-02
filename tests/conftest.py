"""pytest 共享配置:确保项目根在 sys.path 上,便于 import research_assistant。"""

import sys
from pathlib import Path

# 项目根目录(本文件的上上级),使 tests 无需 pip install 也能导入包
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
