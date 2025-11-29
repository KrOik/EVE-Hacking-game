import random
import json
import multiprocessing
import time
import os
import sys
from copy import deepcopy

# Add path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from game.autopilot import Config
from game.models import System, SnowflakeIDGenerator
from game.autopilot import AutoPilot
from simulate import run_single_game as _run_single_game_sim

# --- 遗传算法配置 ---
POPULATION_SIZE = 24        # 种群大小：每一代同时保留的个体（策略）数量
GENERATIONS = 45            # 进化总代数：算法循环优化多少次
GAMES_PER_EVAL = 550        # 训练集地图数量：每代评估每个个体时使用的随机地图种子数（越多越准但越慢）
VALIDATION_GAMES = 60       # 验证集地图数量：每代最佳个体额外在全新地图上做泛化测试的地图数，用于检测过拟合
ELITISM_COUNT = 4           # 精英保留数：每代直接保留到下一代的最优个体数，防止优秀基因丢失
# Define the parameter space to optimize
PARAM_SPACE = {
    'DISTANCE_HINT': (0.0, 50.0),
    'UNKNOWN_NEIGHBORS': (0.0, 20.0),
    'UTILITY_LOOT': (0.0, 100.0),
    'ATTACK_CORE': (50.0, 200.0),
    'ATTACK_THREAT': (0.0, 100.0),
    'AVOID_RISK': (-50.0, 0.0),
    'ATTACK_THRESHOLD_HIGH_HP': (0.0, 50.0),
    'ATTACK_THRESHOLD_LOW_HP': (10.0, 80.0),
    'PRIORITY_SUPPRESSOR': (10.0, 100.0),
    'PRIORITY_RESTO': (10.0, 100.0),
    'PRIORITY_ANTIVIRUS': (0.0, 50.0),
    'PRIORITY_FIREWALL': (0.0, 30.0)
}

def run_game_with_seed(args):
    """
    [Wrapper] 使用特定随机种子运行单局游戏。
    
    设计用于多进程环境（multiprocessing），因为Pool.map只能接受单个参数。
    
    Args:
        args (tuple): 包含两个元素的元组 (genes, seed)
            - genes (dict): Autopilot的配置参数字典
            - seed (str/int): 地图生成的随机种子，确保可复现性
            
    Returns:
        dict: 包含游戏结果统计信息的字典
            - result (str): 'win' 或 'loss'
            - turns (int): 消耗的回合数
            - health (float): 病毒剩余健康值
            - core_health (float): 系统核心剩余健康值
    """
    genes, seed = args
    
    # 应用基因参数到配置
    if genes:
        Config.update(genes)
    
    # 使用显式种子初始化系统，确保环境一致性
    system = System(seed=seed)
    pilot = AutoPilot(system)
    
    turns = 0
    max_turns = 1000  # 防止死循环的安全上限
    while not system.virus.is_dead and not system.core.is_dead:
        turns += 1
        if turns > max_turns:
            break
            
        action_taken = pilot.step()
        if not action_taken:
            break
            
    is_win = system.core.is_dead
    return {
        'result': 'win' if is_win else 'loss',
        'turns': turns,
        'health': system.virus.coherence,
        'core_health': system.core.coherence
    }

class Genome:
    """
    [DNA] 代表遗传算法中的一个个体（策略配置）。
    包含一组参数（基因）及其评估指标（适应度）。
    """
    def __init__(self, genes=None):
        self.genes = genes if genes else self._random_genes()
        self.fitness = 0.0
        self.stats = {}

    def _random_genes(self):
        """
        生成随机基因。
        
        Returns:
            dict: 初始化的随机参数字典
        """
        genes = {}
        for key, (min_val, max_val) in PARAM_SPACE.items():
            genes[key] = random.uniform(min_val, max_val)
        return genes

    def mutate(self, rate=0.1, strength=0.2):
        """
        [变异] 对基因进行随机扰动，引入多样性。
        
        Args:
            rate (float): 变异概率 (0.0 - 1.0)，每个基因发生变异的几率
            strength (float): 变异强度 (0.0 - 1.0)，参数值改变的幅度比例
        """
        for key, (min_val, max_val) in PARAM_SPACE.items():
            if random.random() < rate:
                change = (max_val - min_val) * strength * random.uniform(-1, 1)
                self.genes[key] = max(min_val, min(max_val, self.genes[key] + change))

def evaluate_population_on_seeds(genomes, seeds):
    """
    [评估] 在固定的一组地图种子上评估整个种群的性能。
    使用多进程并行计算以利用多核CPU优势。
    
    Args:
        genomes (list[Genome]): 待评估的基因组列表
        seeds (list): 用于生成地图的随机种子列表
        
    Returns:
        list[dict]: 每个基因组的评估结果列表，包含适应度分数和详细统计
    """
    # 准备任务列表：将每个基因组与每个种子组合
    # 总任务数 = 种群大小 * 评估地图数
    tasks = []
    for i, genome in enumerate(genomes):
        for seed in seeds:
            tasks.append((genome.genes, seed))
    
    # 获取CPU核心数，最大化并行效率
    num_workers = multiprocessing.cpu_count()
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        results_flat = pool.map(run_game_with_seed, tasks)
        
    # 聚合每个基因组的结果
    genome_results = []
    for i in range(len(genomes)):
        start_idx = i * len(seeds)
        end_idx = start_idx + len(seeds)
        subset = results_flat[start_idx:end_idx]
        
        wins = 0
        total_turns = 0
        total_health = 0
        total_core_health = 0
        
        for stats in subset:
            if stats['result'] == 'win':
                wins += 1
                total_health += stats['health']
            total_turns += stats['turns']
            total_core_health += stats['core_health']
            
        win_rate = wins / len(seeds)
        avg_turns = total_turns / len(seeds)
        avg_health = total_health / wins if wins > 0 else 0
        avg_core_health = total_core_health / len(seeds)
        
        # --- 稠密适应度函数设计 (Dense Reward) ---
        # 1. 基础分：胜率 (0-1000分)
        score = (win_rate * 1000.0)
        
        # 2. 速度加成：仅在获胜时给予 (奖励更少回合数)
        if win_rate > 0:
            score += (200.0 / max(1, avg_turns)) * 100.0 
            score += avg_health
        
        # 3. 进度奖励：即使失败，也根据对核心造成的伤害给分 (防止稀疏奖励导致的零梯度)
        # 最大核心血量为90。伤害越高，分数越高。
        # 0 伤害 -> 0 分
        # 90 伤害 (击杀) -> 500 分
        core_damage_pct = (90.0 - avg_core_health) / 90.0
        score += max(0, core_damage_pct) * 500.0
            
        genome_results.append({
            'fitness': score,
            'win_rate': win_rate,
            'avg_turns': avg_turns,
            'avg_health': avg_health,
            'avg_core_health': avg_core_health
        })
        
    return genome_results

class GeneticOptimizer:
    """
    [进化优化器] 管理遗传算法的完整生命周期。
    包括种群初始化、进化循环、选择、交叉、变异以及交叉验证逻辑。
    """
    def __init__(self, resume_from=None):
        """
        初始化优化器。
        
        Args:
            resume_from (str, optional): 包含先前训练好的参数的JSON文件路径。
                                         如果提供，将基于此参数初始化种群（精英策略+局部探索）。
        """
        self.population = [Genome() for _ in range(POPULATION_SIZE)]
        
        if resume_from:
            if os.path.exists(resume_from):
                print(f"Loading starting genome from {resume_from}...")
                try:
                    with open(resume_from, 'r') as f:
                        params = json.load(f)
                    # 策略：将已知的优秀基因注入种群
                    # 1. 精英保留：直接包含完全相同的个体
                    self.population[0] = Genome(genes=params)
                    
                    # 2. 局部探索：基于最优解生成变异体（填充20%种群）
                    # 这有助于在最优解附近寻找更好的局部极值
                    num_mutants = int(POPULATION_SIZE * 0.2)
                    for i in range(1, num_mutants + 1):
                        clone = Genome(genes=params.copy())
                        clone.mutate(rate=0.1, strength=0.2)
                        self.population[i] = clone
                        
                    print(f"Initialized population with saved parameters and {num_mutants} variants.")
                except Exception as e:
                    print(f"Error loading resume file: {e}")
            else:
                print(f"Warning: Resume file {resume_from} not found. Starting from scratch.")

        self.best_genome = None
        # 生成固定种子生成器，确保评估的一致性
        self.id_gen = SnowflakeIDGenerator()
        
        # --- 自适应变异状态 (Adaptive Mutation State) ---
        self.base_mutation_rate = 0.1       # 基础变异率
        self.current_mutation_rate = 0.1    # 当前动态变异率
        self.stagnation_counter = 0         # 停滞计数器（连续多少代未提升）
        self.last_best_fitness = 0.0        # 上一代的最佳适应度

    def evolve(self):
        """
        [核心循环] 执行进化过程。
        
        主要步骤：
        1. 生成训练集种子。
        2. 并行评估种群。
        3. 选择、交叉、变异生成下一代。
        4. 在独立验证集上测试最佳个体（防止过拟合）。
        5. 根据停滞情况动态调整变异率。
        """
        cpu_count = multiprocessing.cpu_count()
        print(f"Starting evolution with Cross-Validation Strategy")
        print(f"Hardware Acceleration: CPU Multiprocessing ({cpu_count} Cores)")
        print(f"Generations: {GENERATIONS}, Pop: {POPULATION_SIZE}")
        print(f"Training Set: {GAMES_PER_EVAL} maps | Validation Set: {VALIDATION_GAMES} maps")
        
        try:
            for gen in range(GENERATIONS):
                print(f"\n--- Generation {gen+1}/{GENERATIONS} ---")
                start_time = time.time()
                
                # 1. 生成训练种子 (每一代都重新生成，防止对特定地图过拟合)
                training_seeds = [self.id_gen.generate_id() for _ in range(GAMES_PER_EVAL)]
                
                # 2. 在训练集上评估种群
                results = evaluate_population_on_seeds(self.population, training_seeds)
                
                # 更新每个个体的适应度
                total_fitness = 0
                best_gen_fitness = -1
                best_idx = 0
                
                for i, res in enumerate(results):
                    self.population[i].fitness = res['fitness']
                    self.population[i].stats = res
                    total_fitness += res['fitness']
                    if res['fitness'] > best_gen_fitness:
                        best_gen_fitness = res['fitness']
                        best_idx = i
                
                # 按适应度降序排序
                self.population.sort(key=lambda x: x.fitness, reverse=True)
                current_best = self.population[0]
                
                # --- 自适应变异逻辑 (Adaptive Mutation) ---
                # 如果适应度有显著提升，重置变异率进行精细搜索
                if current_best.fitness > self.last_best_fitness + 1.0: 
                    self.stagnation_counter = 0
                    self.current_mutation_rate = self.base_mutation_rate 
                else:
                    self.stagnation_counter += 1
                
                # 如果连续停滞，增大变异率以跳出局部最优
                if self.stagnation_counter >= 2:
                    self.current_mutation_rate = min(0.5, self.current_mutation_rate * 1.5)
                    print(f"Stagnation detected ({self.stagnation_counter} gens). Increasing mutation rate to {self.current_mutation_rate:.2f}")
                
                self.last_best_fitness = current_best.fitness

                avg_fitness = total_fitness / POPULATION_SIZE
                print(f"[Train] Best Fitness: {current_best.fitness:.2f} (Win Rate: {current_best.stats['win_rate']*100:.1f}%)")
                print(f"[Train] Avg Fitness: {avg_fitness:.2f}")
                
                # 3. 验证步骤 (Anti-Overfitting Validation)
                # 将最佳个体在一个全新的、更大的验证集上运行
                validation_seeds = [self.id_gen.generate_id() for _ in range(VALIDATION_GAMES)]
                val_results = evaluate_population_on_seeds([current_best], validation_seeds)[0]
                
                print(f"[Valid] Win Rate: {val_results['win_rate']*100:.1f}% | Avg Turns: {val_results['avg_turns']:.1f}")
                
                # 4. 过拟合检测
                # 如果训练胜率显著高于验证胜率，发出警告
                train_win = current_best.stats['win_rate']
                val_win = val_results['win_rate']
                if train_win > val_win + 0.1:
                    print("Warning: Possible Overfitting detected (Train >> Validation)")
                
                self.best_genome = current_best
                
                # [Checkpoint] 每代结束后立即保存最佳参数
                # 这样即使被手动中断（Ctrl+C）也不会丢失这一代的训练成果
                self._save_checkpoint()
                
                print(f"Time: {time.time() - start_time:.2f}s")
                
                if gen < GENERATIONS - 1:
                    self.population = self._next_generation()
            
            print("\nOptimization Complete!")
            
        except KeyboardInterrupt:
            print("\n[Interrupt] Training interrupted by user!")
            print("Saving current best parameters before exiting...")
            self._save_checkpoint()
            
        except Exception as e:
            print(f"\n[Error] Unexpected error during training: {e}")
            if self.best_genome:
                print("Saving current best parameters before crashing...")
                self._save_checkpoint()
            raise e

    def _save_checkpoint(self):
        """保存当前最佳参数到文件"""
        if not self.best_genome:
            return
            
        try:
            # 先写入临时文件，再重命名，防止写入中断导致文件损坏
            temp_file = 'best_autopilot_params.json.tmp'
            target_file = 'best_autopilot_params.json'
            
            with open(temp_file, 'w') as f:
                json.dump(self.best_genome.genes, f, indent=4)
            
            if os.path.exists(target_file):
                os.remove(target_file)
            os.rename(temp_file, target_file)
            
            # 仅在最后一代或中断时打印，避免刷屏
            # print("Checkpoint saved.") 
        except Exception as e:
            print(f"Failed to save checkpoint: {e}")

    def _next_generation(self):
        """
        生成下一代种群。
        采用锦标赛选择策略，结合均匀交叉和变异操作。
        保留精英个体直接进入下一代。
        """
        new_pop = []
        new_pop.extend([deepcopy(g) for g in self.population[:ELITISM_COUNT]])
        
        while len(new_pop) < POPULATION_SIZE:
            parent1 = self._tournament_select()
            parent2 = self._tournament_select()
            
            child_genes = {}
            for key in PARAM_SPACE:
                if random.random() < 0.5:
                    child_genes[key] = parent1.genes[key]
                else:
                    child_genes[key] = parent2.genes[key]
            
            child = Genome(child_genes)
            child.mutate(rate=self.current_mutation_rate)
            new_pop.append(child)
            
        return new_pop

    def _tournament_select(self, k=3):
        """
        [选择] 锦标赛选择法。
        随机选择k个个体，返回其中适应度最高的一个。
        """
        tournament = random.sample(self.population, k)
        tournament.sort(key=lambda x: x.fitness, reverse=True)
        return tournament[0]

if __name__ == "__main__":
    # Windows multiprocessing support
    # 在Windows上，多进程必须放在if __name__ == "__main__":保护块下
    multiprocessing.freeze_support()
    
    import argparse
    parser = argparse.ArgumentParser(description="Train EVE Hacking Autopilot")
    
    # 训练超参数配置 (Training Hyperparameters)
    parser.add_argument('--resume', action='store_true', help='Resume training from best_autopilot_params.json (从上次最佳参数继续训练)')
    parser.add_argument('--generations', type=int, default=GENERATIONS, help=f'Number of generations (进化代数) [default: {GENERATIONS}]')
    parser.add_argument('--pop-size', type=int, default=POPULATION_SIZE, help=f'Population size (种群大小) [default: {POPULATION_SIZE}]')
    parser.add_argument('--games', type=int, default=GAMES_PER_EVAL, help=f'Games per evaluation (训练集地图数/代) [default: {GAMES_PER_EVAL}]')
    parser.add_argument('--val-games', type=int, default=VALIDATION_GAMES, help=f'Validation games (验证集地图数/代) [default: {VALIDATION_GAMES}]')
    parser.add_argument('--elitism', type=int, default=ELITISM_COUNT, help=f'Elitism count (精英保留数) [default: {ELITISM_COUNT}]')
    
    args = parser.parse_args()
    
    # 使用命令行参数更新全局配置
    GENERATIONS = args.generations
    POPULATION_SIZE = args.pop_size
    GAMES_PER_EVAL = args.games
    VALIDATION_GAMES = args.val_games
    ELITISM_COUNT = args.elitism
    
    # 参数合理性检查
    if ELITISM_COUNT >= POPULATION_SIZE:
        print(f"Error: Elitism count ({ELITISM_COUNT}) must be less than population size ({POPULATION_SIZE})")
        sys.exit(1)

    resume_file = 'best_autopilot_params.json' if args.resume else None
    
    print(f"Mode: {'RESUME/FINE-TUNE' if args.resume else 'TRAIN FROM SCRATCH'}")
    print("-" * 30)
    print(f"Training Configuration:")
    print(f"  Generations:      {GENERATIONS}")
    print(f"  Population Size:  {POPULATION_SIZE}")
    print(f"  Training Games:   {GAMES_PER_EVAL}")
    print(f"  Validation Games: {VALIDATION_GAMES}")
    print(f"  Elitism Count:    {ELITISM_COUNT}")
    print("-" * 30)
    
    optimizer = GeneticOptimizer(resume_from=resume_file)
    optimizer.evolve()
