import sys
import argparse
import importlib.util
import os
import json
from pathlib import Path
from typing import Optional


def load_example_module(filepath):
    """Dynamically load a Python module from file."""
    spec = importlib.util.spec_from_file_location("example_module", filepath)
    module = importlib.util.module_from_spec(spec)
    sys.modules["example_module"] = module
    spec.loader.exec_module(module)
    return module


def cmd_trace(args):
    """Execute trace command with enhanced features."""
    from phantom_core import trace, __version__
    from phantom_core.utils.determinism import set_global_determinism, get_determinism_state
    from phantom_core.replay.replay import create_replay_artifact, compute_graph_hash
    from phantom_core.utils.env import get_env_info
    from phantom_core.exec.targets import get_available_targets
    import torch
    
    print(f"🚀 PHANTOM Trace v{__version__}: {args.module}")
    
    # Check target availability
    available_targets = get_available_targets()
    if args.target not in available_targets:
        print(f"❌ Error: Target '{args.target}' not available")
        print(f"   Available targets: {', '.join(available_targets)}")
        return 1
    
    # Load example module
    module = load_example_module(args.module)
    
    if not hasattr(module, 'build_model') or not hasattr(module, 'sample_inputs'):
        print("❌ Error: Module must define build_model() and sample_inputs()")
        return 1
    
    # Build model and inputs
    print("📦 Building model...")
    model = module.build_model()
    inputs = module.sample_inputs()
    
    # Set determinism
    set_global_determinism(args.seed)
    
    # Trace and analyze
    print(f"🔍 Tracing with target={args.target}, abs_tol={args.abs_tol}, rel_tol={args.rel_tol}")
    
    report = trace(
        model,
        inputs,
        target=args.target,
        tolerances={"abs": args.abs_tol, "rel": args.rel_tol},
        seed=args.seed,
        capture_intermediates=args.capture_intermediates,
        use_hooks=args.use_hooks,
        interactive=args.interactive
    )
    
    # Save report
    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else '.', exist_ok=True)
    report.save_html(args.out)
    
    # Create replay artifact
    replay_file = create_replay_artifact(
        seeds=get_determinism_state(),
        env_info=get_env_info(),
        model_ident=args.module,
        graph_hash=compute_graph_hash(report.graph) if report.graph else "unknown",
        tolerances={"abs": args.abs_tol, "rel": args.rel_tol},
        slice_spec=args.slice,
        cli_command=" ".join(sys.argv),
        output_dir="runs"
    )
    
    print(f"📊 Report saved: {args.out}")
    print(f"💾 Replay artifact: {replay_file}")
    
    # Print summary
    if report.first_bad_op is not None:
        print(f"\n⚠️  Drift detected at operation #{report.first_bad_op}")
        print(f"   Max abs error: {report.metrics.get('max_abs_error', 0):.2e}")
        print(f"   Max rel error: {report.metrics.get('max_rel_error', 0):.2e}")
        
        # Print fusion info if available
        if hasattr(report, 'fusion_info') and report.fusion_info:
            print(f"\n⚡ Kernel Fusion Analysis:")
            print(f"   Detected {report.fusion_info['total_opportunities']} fusion opportunities")
            for rec in report.fusion_info.get('recommendations', [])[:2]:
                print(f"   💡 {rec}")
        
        return 1
    else:
        print("\n✅ No significant drift detected")
        print(f"   Max abs error: {report.metrics.get('max_abs_error', 0):.2e}")
        print(f"   Max rel error: {report.metrics.get('max_rel_error', 0):.2e}")
        return 0


def cmd_replay(args):
    """Execute replay command with validation."""
    from phantom_core.replay.replay import load_replay_artifact
    from phantom_core.utils.determinism import set_global_determinism
    import subprocess
    import json
    
    print(f"🔄 PHANTOM Replay: {args.replay_file}")
    
    try:
        # Load replay artifact
        artifact = load_replay_artifact(args.replay_file)
    except FileNotFoundError:
        print(f"❌ Error: Replay file not found: {args.replay_file}")
        return 1
    except json.JSONDecodeError:
        print(f"❌ Error: Invalid replay file format")
        return 1
    
    print(f"📅 Original run: {artifact['timestamp']}")
    print(f"🎯 Model: {artifact['model']['identifier']}")
    print(f"🔧 Original tolerances: abs={artifact['config']['tolerances']['abs']}, "
          f"rel={artifact['config']['tolerances']['rel']}")
    
    # Set determinism from saved state
    if 'torch_seed' in artifact['seeds']:
        set_global_determinism(int(artifact['seeds']['torch_seed']))
        print(f"🎲 Seed restored: {artifact['seeds']['torch_seed']}")
    
    # Validate environment if requested
    if args.validate_env:
        from phantom_core.utils.env import get_env_info
        current_env = get_env_info()
        original_env = artifact['env']
        
        mismatches = []
        for key in ['torch_version', 'cuda_version', 'gpu_name']:
            if key in original_env and original_env[key] != current_env.get(key):
                mismatches.append(f"{key}: {original_env[key]} → {current_env.get(key)}")
        
        if mismatches:
            print("\n⚠️  Environment differences detected:")
            for mismatch in mismatches:
                print(f"   {mismatch}")
            if not args.force:
                print("\n   Use --force to replay anyway")
                return 1
    
    # Re-run the original command
    original_cmd = artifact['cli']
    
    if args.dry_run:
        print(f"\n📋 Would execute: {original_cmd}")
        return 0
    
    print(f"\n🔍 Re-executing original trace...")
    
    # Parse and modify original command for replay
    cmd_parts = original_cmd.split()
    if '--seed' in cmd_parts:
        seed_idx = cmd_parts.index('--seed')
        cmd_parts[seed_idx + 1] = str(artifact['seeds'].get('torch_seed', 1234))
    
    # Execute
    result = subprocess.run(cmd_parts, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ PASS: Replay successful")
    else:
        print(f"❌ FAIL: Replay failed with code {result.returncode}")
        if args.verbose:
            print(f"\nOutput:\n{result.stdout}")
            print(f"\nErrors:\n{result.stderr}")
    
    return result.returncode


def cmd_compare(args):
    """Compare multiple runs (NEW)."""
    from phantom_core.replay.replay import load_replay_artifact
    import json
    from tabulate import tabulate
    
    print(f"📊 PHANTOM Compare: {len(args.runs)} runs")
    
    runs_data = []
    for run_file in args.runs:
        try:
            artifact = load_replay_artifact(run_file)
            runs_data.append({
                'file': os.path.basename(run_file),
                'timestamp': artifact['timestamp'],
                'model': os.path.basename(artifact['model']['identifier']),
                'abs_tol': artifact['config']['tolerances']['abs'],
                'rel_tol': artifact['config']['tolerances']['rel'],
                'env': artifact['env'].get('gpu_name', 'CPU')
            })
        except Exception as e:
            print(f"⚠️  Failed to load {run_file}: {e}")
    
    if not runs_data:
        print("❌ No valid runs to compare")
        return 1
    
    # Display comparison table
    headers = ['Run', 'Model', 'Abs Tol', 'Rel Tol', 'Device']
    table_data = [
        [r['file'][:20], r['model'], f"{r['abs_tol']:.1e}", 
         f"{r['rel_tol']:.1e}", r['env'][:15]]
        for r in runs_data
    ]
    
    print("\n" + tabulate(table_data, headers=headers, tablefmt='grid'))
    
    return 0


def cmd_profile(args):
    """Profile model execution (NEW)."""
    from phantom_core import trace
    from phantom_core.exec.targets import get_available_targets
    import torch
    import time
    
    print(f"⏱️  PHANTOM Profile: {args.module}")
    
    # Load module
    module = load_example_module(args.module)
    model = module.build_model()
    inputs = module.sample_inputs()
    
    # Warmup
    print("🔥 Warming up...")
    for _ in range(3):
        with torch.no_grad():
            _ = model(inputs)
    
    # Profile each backend
    results = {}
    for target in args.targets:
        if target not in get_available_targets():
            print(f"⚠️  Skipping unavailable target: {target}")
            continue
        
        print(f"\n📍 Profiling {target}...")
        
        # Move model and inputs
        if target == 'cuda':
            device = torch.device('cuda')
            model_target = model.to(device)
            inputs_target = inputs.to(device)
        elif target == 'mps':
            device = torch.device('mps')
            model_target = model.to(device)
            inputs_target = inputs.to(device)
        else:
            model_target = model.cpu()
            inputs_target = inputs.cpu()
        
        # Time execution
        times = []
        for _ in range(args.iterations):
            if target in ['cuda', 'mps']:
                torch.cuda.synchronize() if target == 'cuda' else None
            
            start = time.perf_counter()
            with torch.no_grad():
                _ = model_target(inputs_target)
            
            if target in ['cuda', 'mps']:
                torch.cuda.synchronize() if target == 'cuda' else None
            
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms
        
        # Statistics
        times = sorted(times)
        results[target] = {
            'mean': sum(times) / len(times),
            'median': times[len(times) // 2],
            'min': times[0],
            'max': times[-1],
            'p95': times[int(len(times) * 0.95)]
        }
    
    # Display results
    print("\n📊 Profiling Results (ms):")
    print("-" * 60)
    
    from tabulate import tabulate
    headers = ['Target', 'Mean', 'Median', 'Min', 'Max', 'P95']
    table_data = [
        [target, f"{stats['mean']:.2f}", f"{stats['median']:.2f}",
         f"{stats['min']:.2f}", f"{stats['max']:.2f}", f"{stats['p95']:.2f}"]
        for target, stats in results.items()
    ]
    
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    # Find fastest
    fastest = min(results.items(), key=lambda x: x[1]['median'])
    print(f"\n🏆 Fastest: {fastest[0]} (median: {fastest[1]['median']:.2f}ms)")
    
    return 0


def main():
    """Main CLI entry point - Enhanced version."""
    parser = argparse.ArgumentParser(
        description="PHANTOM - Cross-accelerator debugger v0.2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  phantom trace model.py --target=cuda --interactive
  phantom replay runs/2025-08-25.json --validate-env
  phantom compare runs/*.json
  phantom profile model.py --targets cpu cuda mps
        """
    )
    
    # Global options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Trace command (enhanced)
    trace_parser = subparsers.add_parser('trace', help='Trace and compare model execution')
    trace_parser.add_argument('module', help='Python module with model')
    trace_parser.add_argument('--target', default='cuda',
                            choices=['cuda', 'mps', 'rocm', 'cpu'],
                            help='Target backend')
    trace_parser.add_argument('--abs_tol', type=float, default=1e-3,
                            help='Absolute tolerance')
    trace_parser.add_argument('--rel_tol', type=float, default=1e-3,
                            help='Relative tolerance')
    trace_parser.add_argument('--precision', default='fp16',
                            choices=['fp16', 'bf16', 'fp32', 'fp64'],
                            help='Target precision')
    trace_parser.add_argument('--out', default='reports/run.html',
                            help='Output HTML report')
    trace_parser.add_argument('--seed', type=int, default=1234,
                            help='Random seed')
    trace_parser.add_argument('--slice', help='Slice spec for memory efficiency')
    trace_parser.add_argument('--interactive', action='store_true',
                            help='Generate interactive Plotly report')
    trace_parser.add_argument('--capture-intermediates', action='store_true',
                            help='Capture all intermediate activations')
    trace_parser.add_argument('--use-hooks', action='store_true',
                            help='Use hook-based capture for dynamic models')
    
    # Replay command (enhanced)
    replay_parser = subparsers.add_parser('replay', help='Replay a previous run')
    replay_parser.add_argument('replay_file', help='Replay JSON file')
    replay_parser.add_argument('--validate-env', action='store_true',
                             help='Validate environment matches original')
    replay_parser.add_argument('--force', action='store_true',
                             help='Force replay despite environment differences')
    replay_parser.add_argument('--dry-run', action='store_true',
                             help='Show command without executing')
    
    # Compare command (NEW)
    compare_parser = subparsers.add_parser('compare', help='Compare multiple runs')
    compare_parser.add_argument('runs', nargs='+', help='Replay JSON files to compare')
    
    # Profile command (NEW)
    profile_parser = subparsers.add_parser('profile', help='Profile model performance')
    profile_parser.add_argument('module', help='Python module with model')
    profile_parser.add_argument('--targets', nargs='+',
                               default=['cpu', 'cuda'],
                               help='Backends to profile')
    profile_parser.add_argument('--iterations', type=int, default=100,
                               help='Number of iterations')
    
    args = parser.parse_args()
    
    # Setup logging
    from phantom_core.utils.logging import setup_logging
    setup_logging(args.log_level if hasattr(args, 'log_level') else 'INFO')
    
    # Execute command
    if args.command == 'trace':
        return cmd_trace(args)
    elif args.command == 'replay':
        return cmd_replay(args)
    elif args.command == 'compare':
        return cmd_compare(args)
    elif args.command == 'profile':
        return cmd_profile(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
