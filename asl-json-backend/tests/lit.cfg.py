import platform
import lit.formats
from lit.llvm.subst import ToolSubst

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ASLJsonBackendTest import ASLJsonBackendTest

config.name = 'ASL-JSON-BACKEND'
config.test_format = ASLJsonBackendTest(True)
config.suffixes = [ ".asl" ]
config.substitutions = [
    ('%ASL-JSON-BACKEND', config.asl_json_backend),
]
config.test_source_root = os.path.dirname(__file__)