"""Write the Ollama Modelfile for a TinyJev v2 GGUF. The SYSTEM text must be byte-identical to training (convert_tev1.SYSTEM)."""
import sys
from convert_tev1 import SYSTEM
gguf, out = sys.argv[1], sys.argv[2]
open(out, "w").write(f'FROM {gguf}\nSYSTEM """{SYSTEM}"""\nPARAMETER num_ctx 16384\nCAPABILITY decision\nREQUIRES 0.35.1\n')
print(open(out).read())
