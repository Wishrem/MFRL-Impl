import pytest
from test.utils import is_same_array
from envs.grid_world import GridWorldEnv, ForbiddenAreaCfg, RewardCfg

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

class TestGridWorldEnv:
   
    def test_init(self):
        env = GridWorldEnv(start_loc=(0, 0), target_loc=(1, 1), size=5, forbidden_area_cfg=ForbiddenAreaCfg(locs=[(0, 0)]))
        assert is_same_array(env._target_loc, (1, 1), True)
        assert env.size == (5, 5)
        assert is_same_array(env._forbidden_locs, [(0, 0)])
        assert is_same_array(env._start_loc, (0, 0), True)
        assert is_same_array(env._agent_loc, (0, 0), True)
        
        env = GridWorldEnv(start_loc=(0, 0), target_loc=None, size=(2, 2), forbidden_area_cfg=ForbiddenAreaCfg(locs=[(0, 0), (0, 1), (1, 0)]))
        assert is_same_array(env._target_loc, (1, 1), True)
        assert env.size == (2, 2)
        assert is_same_array(env._forbidden_locs, [(0, 0), (0, 1), (1, 0)])
        assert is_same_array(env._start_loc, (0, 0), True)
        assert is_same_array(env._agent_loc, (0, 0), True)
        
        # target loc overlaps the forbidden locs
        with pytest.raises(ValueError):
            env = GridWorldEnv(start_loc=(0, 0), target_loc=(0, 0), size=5, forbidden_area_cfg=ForbiddenAreaCfg(locs=[(0, 0)]))
        
        # the nubmer of forbidden locs is greater than or equal to the number of grid world locations
        with pytest.raises(ValueError):
            env = GridWorldEnv(start_loc=(0, 0), target_loc=(1, 1), size=2, forbidden_area_cfg=ForbiddenAreaCfg(num=4)) 
            
    def test_step(self):
        env = GridWorldEnv(start_loc=(0, 0), target_loc=(1, 1), size=2,
                           forbidden_area_cfg=ForbiddenAreaCfg(locs=[(1, 0)]),
                           reward_cfg=RewardCfg(out_of_bound=-2, forbidden_area=-1, target=1, move=0))
        
        # move out of bound
        _, rwd, *_ = env.step(3) # left
        assert rwd == -2
        assert is_same_array(env._agent_loc, (0, 0), True)
        
        # move to a normal area
        _, rwd, *_ = env.step(1) # right
        assert rwd == 0
        assert is_same_array(env._agent_loc, (0, 1), True)
        
        # stay
        _, rwd, *_ = env.step(4) # stay
        assert rwd == 0
        assert is_same_array(env._agent_loc, (0, 1), True)
        
        # move to target
        _, rwd, *_ = env.step(2) # down
        assert rwd == 1
        assert is_same_array(env._agent_loc, (1, 1), True)
        
         # stay
        _, rwd, *_ = env.step(4) # stay
        assert rwd == 1
        assert is_same_array(env._agent_loc, (1, 1), True)
        
        # move to forbidden area
        _, rwd, *_ = env.step(3) # left
        assert rwd == -1
        assert is_same_array(env._agent_loc, (1, 0), True)
        
         # stay
        _, rwd, *_ = env.step(4) # stay
        assert rwd == -1
        assert is_same_array(env._agent_loc, (1, 0), True)
        
    def test_reset(self):
        env = GridWorldEnv(start_loc=(0, 0), target_loc=(1, 1), size=2,
                           forbidden_area_cfg=ForbiddenAreaCfg(locs=[(1, 0)]),
                           reward_cfg=RewardCfg(out_of_bound=-2, forbidden_area=-1, target=1, move=0))
        
        obs, *_ = env.step(1) # right
        assert is_same_array(obs["agent_loc"], (0, 1), True)
        obs, _ = env.reset()
        assert is_same_array(obs["agent_loc"], (0, 0), True)
        
        # with option: agent_loc
        obs, _ =env.reset(options={
            "agent_loc": (1, 1)
        })
        assert is_same_array(obs["agent_loc"], (1, 1), True)