import ast
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class TrainingPortabilityTests(unittest.TestCase):
    def test_import_and_spawn_workers_do_not_start_training(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as td:
            script = Path(td)/'spawn_check.py'
            script.write_text("""import train
import torch
from utils.multi_epoch_dataloader import MultiEpochsDataLoader
if __name__ == '__main__':
    dataset = torch.utils.data.TensorDataset(torch.arange(4))
    loader = MultiEpochsDataLoader(dataset, batch_size=2, num_workers=2, multiprocessing_context='spawn')
    assert sum(len(batch[0]) for batch in loader) == 4
    print('import and spawn passed')
""", encoding='utf-8')
            env = os.environ.copy()
            env['PYTHONPATH'] = str(root) + os.pathsep + env.get('PYTHONPATH','')
            result = subprocess.run([sys.executable,str(script)],cwd=td,env=env,
                                    capture_output=True,text=True,timeout=90)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('import and spawn passed',result.stdout)
            self.assertFalse((Path(td)/'save').exists())
            self.assertFalse((Path(td)/'log').exists())

    def test_no_nvidia_monitoring_dependency(self):
        root = Path(__file__).resolve().parents[1]
        tree = ast.parse((root/'train.py').read_text(encoding='utf-8'))
        imports = {name.name for n in ast.walk(tree) if isinstance(n,ast.Import) for name in n.names}
        self.assertNotIn('pynvml',imports)


if __name__ == '__main__':
    unittest.main()
