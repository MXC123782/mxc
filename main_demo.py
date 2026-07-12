"""
无线信道竞争均衡策略 — Demo 主程序
=====================================================
基于 IEEE 论文: "Research on Equilibrium Strategy of Wireless Channel Competition"
        Author: Xiangcheng Meng

运行方式:
  python main_demo.py          → 完整演示
  python main_demo.py --quick  → 快速模式 (仅值迭代)
  python main_demo.py --sim N  → 自定义设备数仿真
"""

import os
import sys
import io
import argparse
import numpy as np

# 修复 Windows GBK 编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from typing import Dict

# 确保在正确的目录运行
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from stochastic_game import (
    ChannelState, DeviceAction,
    STATE_NAMES, ACTION_NAMES,
    N_STATES, N_ACTIONS,
    ValueIterator, ValueIterationResult,
    MultiDeviceConfig, MultiDeviceSimulator,
    explain_policy, compare_policies,
    mean_field_approximation,
)
from visualization import (
    plot_convergence, plot_q_values,
    plot_multi_device, plot_decision_boundary,
    plot_system_architecture,
)

# 输出目录
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ╔══════════════════════════════════════════════════════════╗
# ║  Phase 1: 值迭代求解 (论文第 IV 节核心)                 ║
# ╚══════════════════════════════════════════════════════════╝

def phase1_value_iteration() -> ValueIterationResult:
    """阶段 1: 运行值迭代算法, 求解纳什均衡策略"""
    
    print("\n" + "=" * 65)
    print("  阶段 1: 随机博弈值迭代求解")
    print("  论文 §IV — Value Iteration Algorithm")
    print("=" * 65)
    
    # ── 参数 (论文默认值) ──
    GAMMA = 0.9
    EPSILON = 1e-4
    
    print(f"\n  📐 算法参数:")
    print(f"     状态空间: {[STATE_NAMES[s] for s in range(N_STATES)]}")
    print(f"     动作空间: {[ACTION_NAMES[a] for a in range(N_ACTIONS)]}")
    print(f"     折扣因子 γ: {GAMMA}")
    print(f"     收敛阈值 ε: {EPSILON}")
    print(f"     对手策略: 均匀随机 (P(Transmit)=P(Wait)=0.5)")
    
    print(f"\n  📊 奖励函数 R(s, a1, a2):")
    print(f"     ┌──────────┬──────────┬──────────┬──────────┐")
    print(f"     │  状态     │ 本地动作  │ 对手动作  │  奖励    │")
    print(f"     ├──────────┼──────────┼──────────┼──────────┤")
    print(f"     │ Idle     │ Transmit │ Wait     │  +2.0 ✓  │")
    print(f"     │ Idle     │ Transmit │ Transmit │  -1.0 ✗  │")
    print(f"     │ Idle     │ Wait     │ *        │   0.0    │")
    print(f"     │ Congested│ Transmit │ *        │  -1.0 ✗  │")
    print(f"     │ Congested│ Wait     │ *        │   0.0    │")
    print(f"     └──────────┴──────────┴──────────┴──────────┘")
    
    # ── 求解 ──
    print(f"\n  ⏳ 运行值迭代...")
    solver = ValueIterator(gamma=GAMMA, epsilon=EPSILON)
    result = solver.solve(verbose=True)
    
    # ── 输出结果 ──
    print(explain_policy(result))
    
    # ── 验证收敛 ──
    print(f"\n  ✅ 收敛验证:")
    print(f"     迭代次数: {result.iterations}")
    print(f"     最终 δ:   {result.deltas[-1]:.6e}" if result.deltas else "     N/A")
    print(f"     收敛成功: {'✅ 是' if result.converged else '❌ 否'}")
    
    # 期望策略与实际一致
    expected = {ChannelState.IDLE: DeviceAction.TRANSMIT,
                ChannelState.CONGESTED: DeviceAction.WAIT}
    match = all(result.policy[s] == expected[s] for s in expected)
    print(f"     与理论预期一致: {'✅ 是' if match else '⚠ 需检查'}")
    
    return result


# ╔══════════════════════════════════════════════════════════╗
# ║  Phase 2: 多设备信道竞争仿真                             ║
# ╚══════════════════════════════════════════════════════════╝

def phase2_multi_device_sim(n_devices: int = 10):
    """阶段 2: N 设备信道竞争 Monte Carlo 仿真"""
    
    print("\n" + "=" * 65)
    print("  阶段 2: 多设备信道竞争仿真")
    print(f"  设备数: {n_devices} | 时隙数: 1000 | 包到达率: 0.3")
    print("=" * 65)
    
    sim_config = MultiDeviceConfig(
        n_devices=n_devices,
        n_slots=1000,
        packet_arrival_prob=0.3,
        collision_backoff=5,
        noise_prob=0.05,   # 5% 信道噪声
    )
    
    print(f"\n  ⏳ 运行仿真 (3 次 Monte Carlo)...")
    
    # 对博弈策略运行 3 次取平均
    nash_results = []
    for i in range(3):
        sim = MultiDeviceSimulator(sim_config)
        policy = {0: DeviceAction.TRANSMIT, 1: DeviceAction.WAIT}
        r = sim.run(policy)
        nash_results.append(r)
    
    avg_throughput = np.mean([r["throughput"] for r in nash_results])
    avg_collision = np.mean([r["collision_rate"] for r in nash_results])
    avg_jain = np.mean([r["jain_fairness"] for r in nash_results])
    
    print(f"\n  📊 博弈均衡策略 (Nash EQ) 仿真结果:")
    print(f"     平均吞吐量:  {avg_throughput:.4f} 包/时隙")
    print(f"     平均碰撞率:  {avg_collision:.4f}")
    print(f"     Jain 公平性: {avg_jain:.4f}")
    
    print(f"\n  📊 各策略对比:")
    print(f"     ┌─────────────────────┬────────────┬────────────┬────────────┐")
    print(f"     │ 策略                 │ 吞吐量     │ 碰撞率     │ 公平性     │")
    print(f"     ├─────────────────────┼────────────┼────────────┼────────────┤")
    
    # 策略对比
    comp = compare_policies(
        ValueIterationResult(
            V=np.array([0.0, 0.0]),
            policy={0: DeviceAction.TRANSMIT, 1: DeviceAction.WAIT},
            Q=np.zeros((2, 2)),
            history=[], deltas=[], iterations=0, converged=True
        ),
        sim_config
    )
    
    for name, r in comp.items():
        print(f"     │ {name:19s} │ {r['throughput']:+.4f}     │ {r['collision_rate']:+.4f}     │ {r['jain_fairness']:+.4f}     │")
    print(f"     └─────────────────────┴────────────┴────────────┴────────────┘")
    
    return sim_config, comp


# ╔══════════════════════════════════════════════════════════╗
# ║  Phase 3: 高级分析 — 平均场博弈 & 决策边界              ║
# ╚══════════════════════════════════════════════════════════╝

def phase3_advanced_analysis(n_devices: int):
    """阶段 3: 平均场博弈近似 & 决策边界"""
    
    print("\n" + "=" * 65)
    print("  阶段 3: 高级分析")
    print("=" * 65)
    
    # ── 平均场博弈 ──
    print(f"\n  🎯 平均场博弈 (MFG) 近似 (N={n_devices}):")
    print(f"     当 N 很大时, 每个设备将群体平均行为视为环境参数。")
    
    # 计算临界点: 空闲状态下, 若 (N-1) 个设备中每个发送概率 < 1/(N-1),
    # 个体发送的碰撞概率低, 发送更优
    # 简化分析:
    p_critical = 1.0 / max(n_devices - 1, 1)
    print(f"     临界发送概率 (避免碰撞): p < {p_critical:.4f}")
    print(f"     即每个设备发送概率 < {p_critical*100:.1f}% 时, 发送为占优策略")
    
    # 演示
    mf_results = mean_field_approximation(n_devices)
    for state, info in mf_results["决策分析"].items():
        print(f"     {state}: P(对手发送)={info['对手发送概率']:.2f}, "
              f"E[发送]={info['期望收益(发送)']:+.4f}, "
              f"E[等待]={info['期望收益(等待)']:+.4f} → {info['最优动作']}")
    
    print(f"\n     💡 均衡条件: {mf_results['均衡条件']}")
    
    # ── 博弈 vs 最优吞吐量分析 ──
    print(f"\n  🎯 吞吐量与设备密度关系:")
    print(f"     ┌──────────┬────────────────────┬──────────┐")
    print(f"     │ 设备数 N  │ 博弈策略吞吐量     │ 最优退避 │")
    print(f"     ├──────────┼────────────────────┼──────────┤")
    
    for n in [2, 5, 10, 20, 50]:
        cfg = MultiDeviceConfig(n_devices=n, n_slots=500,
                                packet_arrival_prob=0.3, noise_prob=0.05)
        sim = MultiDeviceSimulator(cfg)
        policy = {0: DeviceAction.TRANSMIT, 1: DeviceAction.WAIT}
        r = sim.run(policy)
        optimal_throughput = 1.0 * 0.3  # 理论最优 (无碰撞)
        print(f"     │ {n:8d} │ {r['throughput']:.4f} 包/时隙       │ {optimal_throughput:.4f}     │")
    
    print(f"     └──────────┴────────────────────┴──────────┘")


# ╔══════════════════════════════════════════════════════════╗
# ║  Phase 4: 生成可视化图表                                  ║
# ╚══════════════════════════════════════════════════════════╝

def phase4_visualization(result: ValueIterationResult,
                         sim_config: MultiDeviceConfig,
                         comp_results: Dict):
    """阶段 4: 生成论文风格图表"""
    
    print("\n" + "=" * 65)
    print("  阶段 4: 生成可视化图表")
    print("=" * 65)
    
    print(f"\n  ⏳ 生成图表...")
    
    # 图1: 值迭代收敛曲线
    plot_convergence(result, os.path.join(OUTPUT_DIR, "fig1_convergence.png"))
    
    # 图2: Q 值 & 策略
    plot_q_values(result, os.path.join(OUTPUT_DIR, "fig2_qvalues.png"))
    
    # 图3: 多设备仿真
    plot_multi_device(comp_results, sim_config,
                      os.path.join(OUTPUT_DIR, "fig3_multidevice.png"))
    
    # 图4: 决策边界
    plot_decision_boundary(result,
                           os.path.join(OUTPUT_DIR, "fig4_decision_boundary.png"))
    
    # 图5: 系统架构
    plot_system_architecture(os.path.join(OUTPUT_DIR, "fig5_architecture.png"))
    
    print(f"\n  ✅ 所有图表已保存至: {os.path.abspath(OUTPUT_DIR)}/")
    
    for f in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"     📄 {f} ({size_kb:.1f} KB)")


# ╔══════════════════════════════════════════════════════════╗
# ║  MAIN                                                     ║
# ╚══════════════════════════════════════════════════════════╝

def main():
    parser = argparse.ArgumentParser(
        description="无线信道竞争均衡策略 — Demo 程序",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main_demo.py            完整演示 (值迭代 + 仿真 + 图表)
  python main_demo.py --quick    快速模式 (仅值迭代)
  python main_demo.py --sim 20   20 设备仿真
        """
    )
    parser.add_argument("--quick", action="store_true",
                        help="快速模式 (跳过仿真)")
    parser.add_argument("--sim", type=int, default=10,
                        help="仿真设备数 (默认 10)")
    parser.add_argument("--no-plot", action="store_true",
                        help="跳过图表生成")
    args = parser.parse_args()
    
    print("\n" + "█" * 65)
    print("█" + " " * 63 + "█")
    print("█  无线信道竞争均衡策略 Demo" + " " * 35 + "█")
    print("█  " + "─" * 61 + "█")
    print("█  Based on: \"Research on Equilibrium Strategy of" + " " * 20 + "█")
    print("█            Wireless Channel Competition\"" + " " * 23 + "█")
    print("█  Author:   Xiangcheng Meng" + " " * 36 + "█")
    print("█  Method:   Stochastic Game + Value Iteration" + " " * 22 + "█")
    print("█" + " " * 63 + "█")
    print("█" * 65)
    
    # ── Phase 1: 值迭代 ──
    result = phase1_value_iteration()
    
    if args.quick:
        print(f"\n  🏁 快速模式完成!")
        return
    
    # ── Phase 2: 多设备仿真 ──
    sim_config, comp_results = phase2_multi_device_sim(n_devices=args.sim)
    
    # ── Phase 3: 高级分析 ──
    phase3_advanced_analysis(n_devices=args.sim)
    
    # ── Phase 4: 图表 ──
    if not args.no_plot:
        phase4_visualization(result, sim_config, comp_results)
    
    print(f"\n" + "█" * 65)
    print(f"█  Demo 全部完成! 🎉" + " " * 42 + "█")
    print(f"█" * 65)
    print(f"\n  论文引用模型总结:")
    print(f"  ┌────────────────────────────────────────────────────┐")
    print(f"  │  模型:  两状态-两动作对称随机博弈                   │")
    print(f"  │  算法:  值迭代 (Value Iteration), γ=0.9, ε=1e-4    │")
    print(f"  │  结果:  Idle→Transmit, Congested→Wait              │")
    print(f"  │  意义:  从博弈论角度证明 CSMA/CA 退避策略的最优性  │")
    print(f"  │  扩展:  平均场博弈/DRL/合作博弈/演化博弈            │")
    print(f"  └────────────────────────────────────────────────────┘")

if __name__ == "__main__":
    main()
