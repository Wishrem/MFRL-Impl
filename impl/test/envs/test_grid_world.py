import pytest
from test.utils import is_same_array
from envs.grid_world import GridWorldEnv, ForbiddenAreaCfg

class TestForbiddenAreaCfg:
    grid_size = (2, 2)
    
    def test_init(self):
        # without locs and num
        with pytest.raises(ValueError):
            ForbiddenAreaCfg(locs=None, num=None)
        
        # with locs and num
        locs = [(0, 0), (1, 1)]
        cfg = ForbiddenAreaCfg(locs=locs, num=2)
        assert is_same_array(cfg.locs, locs)
        assert cfg.num == 2
    
    def test_get_locs_with_locs(self):
        locs = [(0, 0), (1, 1)]
        # with num
        cfg = ForbiddenAreaCfg(locs=locs, num=2)
        assert is_same_array(cfg.get_locs(self.grid_size), locs)
        
        # without num
        cfg = ForbiddenAreaCfg(locs=locs, num=None)
        assert is_same_array(cfg.get_locs(self.grid_size), locs)
        
        # out of bound
        locs = [(0, 0), (2, 2)]
        cfg = ForbiddenAreaCfg(locs=locs, num=None)
        with pytest.raises(ValueError):
            cfg.get_locs(self.grid_size)
        
        # full grid
        locs = [(i, j) for i in range(self.grid_size[0]) for j in range(self.grid_size[1])]
        cfg = ForbiddenAreaCfg(locs=locs, num=None)
        with pytest.raises(ValueError):
            cfg.get_locs(self.grid_size)
    
    def test_get_locs_without_locs(self):
        # with num
        cfg = ForbiddenAreaCfg(locs=None, num=2)
        assert len(cfg.get_locs(self.grid_size)) == 2
        
        # full grid
        cfg = ForbiddenAreaCfg(locs=None, num=self.grid_size[0]*self.grid_size[1])
        with pytest.raises(ValueError):
            cfg.get_locs(self.grid_size)
