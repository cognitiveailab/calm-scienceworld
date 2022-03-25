import time

from scienceworld import ScienceWorldEnv


class CALMScienceWorldEnv:
    ''' Returns valid actions at each step of the game. '''

    def __init__(self, rom_path, seed, task_num, step_limit, stuck_step_limit, get_valid=False, simplification_str='easy', thread_offset=0):
        self.env = None
        self.rom_path = rom_path
        self.task_num = task_num
        self.seed = seed
        self.steps = 0
        self.step_limit = step_limit
        self.stuck_step_limit = stuck_step_limit
        self.get_valid = get_valid
        self.max_score = 0
        self.end_scores = []
        self.simplification_str = simplification_str
        self.thread_offset = thread_offset
        self.variation = None

    def load(self, variation_idx):
        self.env.load(self.task_name, variation_idx, self.simplification_str)
        self.variation = variation_idx

    def create(self, thread_id, var_no):
        self.env = ScienceWorldEnv("", self.rom_path, envStepLimit=self.step_limit, threadNum=self.thread_offset+thread_id)
        time.sleep(2)

        taskNames = self.env.getTaskNames()
        self.task_name = taskNames[self.task_num]
        self.env.load(self.task_name, var_no, self.simplification_str)


    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        task_description = self.env.getTaskDescription()
        info['task_description'] = task_description
        ob = ob.replace("\t", " ")

        look = self.env.look()
        info['look'] = look
        inv = self.env.inventory()
        info['inv'] = inv
        info['history'] = self.env.getRunHistory()
        # Get the valid actions for this state
        info['valid'] = [action['action'] for action in self.env.getPossibleActionObjectCombinations()[0]]
        self.steps += 1
        if self.steps >= self.stuck_step_limit:
            done = True
        if info['score'] == -100:
            done = True
        self.max_score = max(self.max_score, info['score'])
        if done: self.end_scores.append(info['score'])
        return ob, reward, done, info

    def reset(self):
        initial_ob, info = self.env.reset()
        task_description = self.env.getTaskDescription()
        info['task_description'] = task_description
        ob = initial_ob
        ob = ob.replace("\t", " ")
        look = self.env.look()
        info['look'] = look
        inv = self.env.inventory()
        info['inv'] = inv
        info['valid'] = [action['action'] for action in self.env.getPossibleActionObjectCombinations()[0]]
        self.steps = 0
        self.max_score = 0
        return ob, info

    def get_end_scores(self, last=1):
        last = min(last, len(self.end_scores))
        return sum(self.end_scores[-last:]) / last if last else 0

    def getVariationsTrain(self):
        return self.env.getVariationsTrain()

    def getVariationsDev(self):
        return self.env.getVariationsDev()
    
    def getVariationsTest(self):
        return self.env.getVariationsTest()

    def close(self):
        self.env.shutdown()
