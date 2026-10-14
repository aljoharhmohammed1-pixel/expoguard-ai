"""Request file -> image gate -> analyzer -> JSON and exit status."""
from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from ai.core.analysis import VisionAnalysis
from ai.inference.analyze_image import main
from ai.inference.fire_adapter import FireAnalyzer
from test_analysis import request, picture, Model, box


class CliTests(unittest.TestCase):
    def invoke(self, metadata, data, model, missing_image=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'metadata.json').write_text(metadata, encoding='utf-8-sig')
            if not missing_image:
                (root / 'image.png').write_bytes(data)
            stdout, stderr = io.StringIO(), io.StringIO()
            args = ['analyze_image', '--metadata', str(root/'metadata.json'), '--image', str(root/'image.png')]
            with patch.object(sys, 'argv', args), redirect_stdout(stdout), redirect_stderr(stderr), \
                 patch('ai.inference.analyze_image.FireAnalyzer', return_value=FireAnalyzer(model)):
                try:
                    code = main()
                except SystemExit as exc:
                    code = exc.code
            return code, stdout.getvalue(), stderr.getvalue()

    def test_success_serializes_contract_and_identity(self):
        code, output, error = self.invoke(request().model_dump_json(), picture(), Model([box()]))
        self.assertEqual(code, 0)
        self.assertEqual(error, '')
        analysis = VisionAnalysis.model_validate_json(output)
        self.assertEqual(analysis.requestId, 'req-1')
        self.assertEqual(analysis.candidates[0].category, 'FIRE')

    def test_unusable_and_inference_error_exit_one(self):
        for data, model, status in [(b'corrupt', Model(), 'UNUSABLE'), (picture(), Model(fail=True), 'ERROR')]:
            code, output, _ = self.invoke(request().model_dump_json(), data, model)
            self.assertEqual(code, 1)
            self.assertEqual(VisionAnalysis.model_validate_json(output).analysisStatus, status)

    def test_invalid_metadata_and_missing_file_exit_two_without_result(self):
        for metadata, missing in [('{bad json', False), ('{}', False), (request().model_dump_json(), True)]:
            code, output, error = self.invoke(metadata, picture(), Model(), missing)
            self.assertEqual(code, 2)
            self.assertEqual(output, '')
            self.assertIn('error:', error)


if __name__ == '__main__':
    unittest.main()
