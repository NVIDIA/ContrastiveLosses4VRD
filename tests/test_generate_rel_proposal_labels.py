import sys
import types
import unittest

class _FakeArray:
    def __getitem__(self, item):
        return [1.0]

class _FakeData:
    def numpy(self):
        return _FakeArray()

class TestGenerateRelProposalLabelsOp(unittest.TestCase):
    def setUp(self):
        nn = types.ModuleType('torch.nn')
        nn.Module = object
        torch = types.ModuleType('torch')
        torch.nn = nn
        sys.modules['torch'] = torch
        sys.modules['torch.nn'] = nn

        model = types.SimpleNamespace(
            USE_SPATIAL_FEAT=False,
            USE_FREQ_BIAS=True,
            USE_NODE_CONTRASTIVE_LOSS=False,
            USE_NODE_CONTRASTIVE_SO_AWARE_LOSS=False,
            USE_NODE_CONTRASTIVE_P_AWARE_LOSS=False,
        )
        cfg = types.SimpleNamespace(MODEL=model)
        core_config = types.ModuleType('core.config')
        core_config.cfg = cfg
        sys.modules['core.config'] = core_config

        json_dataset_rel = types.ModuleType('datasets_rel.json_dataset_rel')
        json_dataset_rel.add_rel_proposals = lambda *a, **k: None
        datasets_rel = types.ModuleType('datasets_rel')
        datasets_rel.json_dataset_rel = json_dataset_rel
        sys.modules['datasets_rel'] = datasets_rel
        sys.modules['datasets_rel.json_dataset_rel'] = json_dataset_rel

        fast_rcnn_rel = types.ModuleType('roi_data_rel.fast_rcnn_rel')
        fast_rcnn_rel.add_rel_blobs = lambda blobs, im_scales, roidb: None
        roi_data_rel = types.ModuleType('roi_data_rel')
        roi_data_rel.fast_rcnn_rel = fast_rcnn_rel
        sys.modules['roi_data_rel'] = roi_data_rel
        sys.modules['roi_data_rel.fast_rcnn_rel'] = fast_rcnn_rel

    def test_freq_bias_blob_names(self):
        from lib.modeling_rel.generate_rel_proposal_labels import GenerateRelProposalLabelsOp
        op = GenerateRelProposalLabelsOp()
        im_info = types.SimpleNamespace(data=_FakeData())
        blobs = op(None, None, None, [], im_info)
