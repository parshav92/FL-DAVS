"""
Attack Logging and Analysis for DAVS+PBFT System
Comprehensive tracking of DAVS scores, committee selection, and attack outcomes
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
from pathlib import Path


class AttackLogger:
    """
    Logger for tracking DAVS+PBFT performance under attacks
    """
    
    def __init__(
        self,
        experiment_name: str,
        save_dir: str,
        malicious_nodes: List[int],
        total_nodes: int
    ):
        """
        Args:
            experiment_name: Name of the experiment
            save_dir: Directory to save logs and plots
            malicious_nodes: List of malicious node IDs
            total_nodes: Total number of nodes in the system
        """
        self.experiment_name = experiment_name
        self.save_dir = save_dir
        self.malicious_nodes = set(malicious_nodes)
        self.honest_nodes = set(range(total_nodes)) - self.malicious_nodes
        self.total_nodes = total_nodes
        
        # Create experiment directory
        self.exp_dir = os.path.join(save_dir, experiment_name)
        Path(self.exp_dir).mkdir(parents=True, exist_ok=True)
        
        # Storage for round-by-round data
        self.rounds_data = []
        
        print(f"📊 Attack Logger initialized")
        print(f"   Experiment: {experiment_name}")
        print(f"   Malicious nodes: {sorted(list(self.malicious_nodes))}")
        print(f"   Honest nodes: {len(self.honest_nodes)}/{total_nodes}")
    
    def log_round(
        self,
        round_num: int,
        davs_scores: Dict[int, float],
        grad_norms: Dict[int, float],
        committee: List[int],
        consensus_result: Dict[str, Any],
        train_loss: float,
        train_acc: float,
        test_loss: Optional[float] = None,
        test_acc: Optional[float] = None,
        blockchain_hash: Optional[str] = None
    ):
        """
        Log data for a single round
        
        Args:
            round_num: Round number
            davs_scores: {node_id: representativeness_score} for ALL nodes
            grad_norms: {node_id: gradient_l2_norm} for ALL nodes
            committee: List of selected committee member IDs
            consensus_result: PBFT consensus details
            train_loss: Training loss
            train_acc: Training accuracy
            test_loss: Optional test loss
            test_acc: Optional test accuracy
            blockchain_hash: Optional block hash
        """
        # Analyze committee composition
        committee_set = set(committee)
        malicious_in_committee = len(committee_set & self.malicious_nodes)
        honest_in_committee = len(committee_set & self.honest_nodes)
        
        # Calculate score statistics
        honest_scores = [davs_scores[nid] for nid in self.honest_nodes if nid in davs_scores]
        malicious_scores = [davs_scores[nid] for nid in self.malicious_nodes if nid in davs_scores]
        
        honest_avg = np.mean(honest_scores) if honest_scores else 0.0
        honest_std = np.std(honest_scores) if honest_scores else 0.0
        malicious_avg = np.mean(malicious_scores) if malicious_scores else 0.0
        malicious_std = np.std(malicious_scores) if malicious_scores else 0.0
        
        score_separation = honest_avg - malicious_avg
        
        # Determine if attack succeeded (malicious node in committee and consensus reached)
        attack_success = (malicious_in_committee > 0 and consensus_result.get('reached', False))
        
        # Create round entry
        round_data = {
            'round': round_num,
            'davs_scores': {int(k): float(v) for k, v in davs_scores.items()},
            'grad_norms': {int(k): float(v) for k, v in grad_norms.items()},
            'committee': committee,
            'committee_size': len(committee),
            'malicious_in_committee': malicious_in_committee,
            'honest_in_committee': honest_in_committee,
            'malicious_selection_rate': malicious_in_committee / len(committee) if committee else 0,
            'honest_avg_score': float(honest_avg),
            'honest_std_score': float(honest_std),
            'malicious_avg_score': float(malicious_avg),
            'malicious_std_score': float(malicious_std),
            'score_separation': float(score_separation),
            'consensus': consensus_result,
            'attack_success': attack_success,
            'metrics': {
                'train_loss': float(train_loss),
                'train_acc': float(train_acc),
                'test_loss': float(test_loss) if test_loss is not None else None,
                'test_acc': float(test_acc) if test_acc is not None else None
            }
        }
        
        if blockchain_hash:
            round_data['blockchain_hash'] = blockchain_hash
        
        self.rounds_data.append(round_data)
        
        # Print summary
        print(f"\n📊 Round {round_num} Summary:")
        print(f"   Committee: {committee}")
        print(f"   Malicious in committee: {malicious_in_committee}/{len(committee)}")
        print(f"   DAVS Score Separation: {score_separation:.4f}")
        print(f"   Honest avg: {honest_avg:.4f} ± {honest_std:.4f}")
        print(f"   Malicious avg: {malicious_avg:.4f} ± {malicious_std:.4f}")
        print(f"   Consensus: {'✅ REACHED' if consensus_result.get('reached') else '❌ FAILED'}")
        print(f"   Attack success: {'❌ YES' if attack_success else '✅ NO (Blocked)'}")
    
    def save_json(self):
        """Save all logged data to JSON"""
        filepath = os.path.join(self.exp_dir, 'attack_log.json')
        
        data = {
            'experiment': self.experiment_name,
            'config': {
                'total_nodes': self.total_nodes,
                'malicious_nodes': sorted(list(self.malicious_nodes)),
                'honest_nodes': sorted(list(self.honest_nodes)),
                'malicious_ratio': len(self.malicious_nodes) / self.total_nodes
            },
            'rounds': self.rounds_data
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Attack log saved to {filepath}")
    
    def plot_davs_score_distribution(self):
        """Plot DAVS score distribution: honest vs malicious"""
        if not self.rounds_data:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'DAVS Score Analysis - {self.experiment_name}', fontsize=14, fontweight='bold')
        
        # Extract data
        rounds = [r['round'] for r in self.rounds_data]
        honest_avgs = [r['honest_avg_score'] for r in self.rounds_data]
        malicious_avgs = [r['malicious_avg_score'] for r in self.rounds_data]
        separations = [r['score_separation'] for r in self.rounds_data]
        
        # Plot 1: Score trends over time
        ax1 = axes[0, 0]
        ax1.plot(rounds, honest_avgs, 'g-o', label='Honest avg', linewidth=2, markersize=4)
        ax1.plot(rounds, malicious_avgs, 'r-x', label='Malicious avg', linewidth=2, markersize=4)
        ax1.fill_between(rounds, honest_avgs, malicious_avgs, alpha=0.3)
        ax1.set_xlabel('Round')
        ax1.set_ylabel('DAVS Score')
        ax1.set_title('DAVS Scores: Honest vs Malicious')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Score separation over time
        ax2 = axes[0, 1]
        ax2.plot(rounds, separations, 'b-o', linewidth=2, markersize=4)
        ax2.axhline(y=0, color='r', linestyle='--', label='No separation')
        ax2.fill_between(rounds, 0, separations, where=[s > 0 for s in separations],
                         color='green', alpha=0.3, label='Positive separation')
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Score Separation (Honest - Malicious)')
        ax2.set_title('DAVS Score Separation')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Box plot of scores (last round)
        ax3 = axes[1, 0]
        last_round = self.rounds_data[-1]
        # Handle both int and string keys (JSON converts int keys to strings)
        davs_scores = last_round['davs_scores']
        honest_scores = [davs_scores.get(nid, davs_scores.get(str(nid), 0)) for nid in self.honest_nodes]
        malicious_scores = [davs_scores.get(nid, davs_scores.get(str(nid), 0)) for nid in self.malicious_nodes]
        
        ax3.boxplot([honest_scores, malicious_scores],
                    labels=['Honest', 'Malicious'],
                    patch_artist=True,
                    boxprops=dict(facecolor='lightblue'))
        ax3.set_ylabel('DAVS Score')
        ax3.set_title(f'Score Distribution (Round {last_round["round"]})')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Gradient Norms
        ax4 = axes[1, 1]
        
        # Handle both int and string keys
        honest_norms_per_round = []
        malicious_norms_per_round = []
        for r in self.rounds_data:
            grad_norms = r['grad_norms']
            honest_norms = [grad_norms.get(nid, grad_norms.get(str(nid), 0)) for nid in self.honest_nodes]
            malicious_norms = [grad_norms.get(nid, grad_norms.get(str(nid), 0)) for nid in self.malicious_nodes]
            honest_norms_per_round.append(np.mean(honest_norms))
            malicious_norms_per_round.append(np.mean(malicious_norms))
        
        ax4.plot(rounds, honest_norms_per_round, 'g-o', label='Honest Avg Norm', linewidth=2, markersize=4)
        ax4.plot(rounds, malicious_norms_per_round, 'r-x', label='Malicious Avg Norm', linewidth=2, markersize=4)
        ax4.set_xlabel('Round')
        ax4.set_ylabel('Gradient L2 Norm')
        ax4.set_title('Gradient Norms: Honest vs Malicious')
        ax4.set_yscale('log')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = os.path.join(self.exp_dir, 'davs_score_analysis.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ DAVS score analysis plot saved to {filepath}")
    
    def plot_accuracy_trends(self):
        """Plot accuracy trends under attack"""
        if not self.rounds_data:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'Model Performance Under Attack - {self.experiment_name}', fontsize=14, fontweight='bold')
        
        rounds = [r['round'] for r in self.rounds_data]
        train_accs = [r['metrics']['train_acc'] for r in self.rounds_data]
        test_accs = [r['metrics']['test_acc'] for r in self.rounds_data if r['metrics']['test_acc'] is not None]
        test_rounds = [r['round'] for r in self.rounds_data if r['metrics']['test_acc'] is not None]
        
        # Plot 1: Accuracy trends
        ax1.plot(rounds, train_accs, 'b-o', label='Train Accuracy', linewidth=2, markersize=4)
        if test_accs:
            ax1.plot(test_rounds, test_accs, 'g-s', label='Test Accuracy', linewidth=2, markersize=4)
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Model Accuracy Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Attack success events
        attack_successes = [r['attack_success'] for r in self.rounds_data]
        consensus_reached = [r['consensus']['reached'] for r in self.rounds_data]
        
        ax2.scatter(rounds, train_accs, c=['red' if s else 'green' for s in attack_successes],
                   s=100, alpha=0.6, edgecolors='black')
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Train Accuracy (%)')
        ax2.set_title('Attack Success Events (Red = Attack succeeded)')
        ax2.grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', edgecolor='black', label='Attack Blocked'),
            Patch(facecolor='red', edgecolor='black', label='Attack Succeeded')
        ]
        ax2.legend(handles=legend_elements)
        
        plt.tight_layout()
        filepath = os.path.join(self.exp_dir, 'accuracy_under_attack.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Accuracy trends plot saved to {filepath}")
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive summary report
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.rounds_data:
            return {}
        
        # Calculate statistics
        total_rounds = len(self.rounds_data)
        
        malicious_selections = sum(r['malicious_in_committee'] for r in self.rounds_data)
        total_selections = sum(r['committee_size'] for r in self.rounds_data)
        
        attack_successes = sum(1 for r in self.rounds_data if r['attack_success'])
        consensus_failures = sum(1 for r in self.rounds_data if not r['consensus']['reached'])
        
        avg_score_separation = np.mean([r['score_separation'] for r in self.rounds_data])
        avg_honest_score = np.mean([r['honest_avg_score'] for r in self.rounds_data])
        avg_malicious_score = np.mean([r['malicious_avg_score'] for r in self.rounds_data])
        
        final_test_acc = next((r['metrics']['test_acc'] for r in reversed(self.rounds_data)
                               if r['metrics']['test_acc'] is not None), None)
        
        summary = {
            'experiment': self.experiment_name,
            'total_rounds': total_rounds,
            'total_nodes': self.total_nodes,
            'malicious_nodes_count': len(self.malicious_nodes),
            'malicious_ratio': len(self.malicious_nodes) / self.total_nodes,
            'performance': {
                'final_test_accuracy': final_test_acc,
                'avg_score_separation': avg_score_separation,
                'avg_honest_score': avg_honest_score,
                'avg_malicious_score': avg_malicious_score
            },
            'attack_resistance': {
                'malicious_selection_rate': malicious_selections / total_selections if total_selections > 0 else 0,
                'random_baseline': len(self.malicious_nodes) / self.total_nodes,
                'attack_success_rate': attack_successes / total_rounds,
                'consensus_failure_rate': consensus_failures / total_rounds
            }
        }
        
        # Save summary
        filepath = os.path.join(self.exp_dir, 'summary_report.json')
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"EXPERIMENT SUMMARY: {self.experiment_name}")
        print(f"{'='*70}")
        print(f"Malicious nodes: {len(self.malicious_nodes)}/{self.total_nodes} ({len(self.malicious_nodes)/self.total_nodes*100:.1f}%)")
        print(f"\nDAVS Performance:")
        print(f"  Avg score separation: {avg_score_separation:.4f}")
        print(f"  Honest avg score: {avg_honest_score:.4f}")
        print(f"  Malicious avg score: {avg_malicious_score:.4f}")
        print(f"\nAttack Resistance:")
        print(f"  Malicious selection rate: {malicious_selections/total_selections*100:.2f}%")
        print(f"  Random baseline: {len(self.malicious_nodes)/self.total_nodes*100:.2f}%")
        print(f"  Attack success rate: {attack_successes/total_rounds*100:.2f}%")
        print(f"  Consensus failure rate: {consensus_failures/total_rounds*100:.2f}%")
        print(f"\nModel Performance:")
        if final_test_acc:
            print(f"  Final test accuracy: {final_test_acc:.2f}%")
        print(f"{'='*70}\n")
        
        return summary
    
    def save_all(self):
        """Save all logs and generate all plots"""
        self.save_json()
        self.plot_davs_score_distribution()
        self.plot_accuracy_trends()
        self.generate_summary_report()
        print(f"✅ All results saved to {self.exp_dir}")
