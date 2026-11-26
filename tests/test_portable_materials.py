from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock
import unittest
from linbo3.comsol import _import_project_materials

class PortableMaterials(unittest.TestCase):
    def test_relative_filenames_after_project_rename(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)/'renamed project';root.mkdir()
            for name in ['n_SiO2.txt','no_LiNbO3.txt','ne_LiNbO3.txt']:(root/name).write_text('1 2')
            model=MagicMock()
            functions={tag:MagicMock() for tag in ['int1','int2','int3']}
            model.java.component.return_value.func.side_effect=lambda tag:functions[tag]
            _import_project_materials(model,root/'model.mph')
            model.java.modelPath.assert_called_once_with(str(root.resolve()))
            for tag,name in zip(functions,['n_SiO2.txt','no_LiNbO3.txt','ne_LiNbO3.txt']):
                functions[tag].set.assert_called_once_with('filename',name)
                functions[tag].importData.assert_called_once_with()
    def test_missing_file_fails_before_model_changes(self):
        with TemporaryDirectory() as tmp:
            model=MagicMock()
            with self.assertRaises(FileNotFoundError):_import_project_materials(model,Path(tmp)/'model.mph')
            model.java.modelPath.assert_not_called()
