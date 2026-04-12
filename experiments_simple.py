import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from simpleCollaborativeClassroom import SimpleCollaborativeClassroom


def get_data_loaders(batch_size=64):
    """Get CIFAR-10 data loaders"""

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    print("Loading CIFAR-10 dataset...")
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )

    # Split into train and validation
    train_size = int(0.8 * len(trainset))
    val_size = len(trainset) - train_size
    trainset, valset = torch.utils.data.random_split(trainset, [train_size, val_size])

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        valset, batch_size=100, shuffle=False, num_workers=2, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2, pin_memory=True
    )

    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")
    print(f"Test samples: {len(testset)}")

    return train_loader, val_loader, test_loader


def run_experiment():
    """Run the collaborative learning experiment"""

    print("=" * 70)
    print("COLLABORATIVE CLASSROOM EXPERIMENT - CIFAR-10")
    print("Testing collective intelligence with 8 students")
    print("=" * 70)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Get data
    train_loader, val_loader, test_loader = get_data_loaders(batch_size=64)

    # Create classroom
    classroom = SimpleCollaborativeClassroom(num_students=8, device=device)
    classroom.set_validation_data(val_loader, num_batches=5)

    # Training
    num_steps = 5000
    eval_every = 200

    print(f"\nTraining for {num_steps} steps...")
    print("=" * 70)

    data_iter = iter(train_loader)
    best_class_acc = 0

    for step in range(num_steps):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)

        classroom.learn_step(x, y, step)

        if step % eval_every == 0 or step == num_steps - 1:
            teacher_acc, best_acc, class_acc, _ = classroom.evaluate_class(test_loader, step)

            advantage = class_acc - best_acc
            symbol = "▲" if advantage > 0 else "▼" if advantage < 0 else "●"

            print(f"Step {step:5d} | Teacher: {teacher_acc:.3f} | "
                  f"Best: {best_acc:.3f} | Class: {class_acc:.3f} | "
                  f"{symbol}{advantage:+.3f} | Phase: {classroom.phase[:5]}")

            if class_acc > best_class_acc:
                best_class_acc = class_acc

    # Results
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    final_class = classroom.history['class_acc'][-1]
    final_best = classroom.history['best_acc'][-1]

    print(f"Final Class Accuracy: {final_class:.4f} ({final_class * 100:.1f}%)")
    print(f"Final Best Student: {final_best:.4f} ({final_best * 100:.1f}%)")
    print(f"Difference: +{(final_class - final_best) * 100:+.1f}%")
    print(f"Teacher usage: {classroom.teacher.get_usage():.1f}%")

    # Check if collective intelligence emerged
    # Look for any point where class outperformed best student
    surpassed = False
    for c, b in zip(classroom.history['class_acc'], classroom.history['best_acc']):
        if c > b:
            surpassed = True
            break

    if surpassed:
        print("\n✅ COLLECTIVE INTELLIGENCE DEMONSTRATED!")
        print("   The class of students outperformed the best individual student")
    else:
        print("\n⚠️  Collective advantage not yet achieved")
        print("   Try running for more steps or increasing num_students")

    return classroom


def plot_results(classroom):
    """Plot the results"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    steps = classroom.history['steps']

    # Plot 1: Accuracy over time
    ax = axes[0]
    ax.plot(steps, classroom.history['best_acc'], 'r-', linewidth=2, label='Best Student')
    ax.plot(steps, classroom.history['class_acc'], 'b-', linewidth=2, label='Class Collective')
    ax.plot(steps, classroom.history['teacher_acc'], 'g--', alpha=0.5, label='Teacher (Oracle)')

    # Mark where class surpasses best
    for i, (c, b) in enumerate(zip(classroom.history['class_acc'],
                                   classroom.history['best_acc'])):
        if c > b:
            ax.axvline(x=steps[i], color='purple', linestyle='--', alpha=0.7, linewidth=2)
            ax.text(steps[i], 0.5, 'Breakthrough', rotation=90, fontsize=10,
                    color='purple', fontweight='bold')
            break

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Collaborative Learning: Class vs Individual', fontsize=12)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 0.8)

    # Plot 2: Improvement over time
    ax = axes[1]
    improvement = [c - b for c, b in zip(classroom.history['class_acc'],
                                         classroom.history['best_acc'])]
    ax.plot(steps, improvement, 'purple', linewidth=2)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=2)

    # Fill areas
    pos_steps = [steps[i] for i in range(len(improvement)) if improvement[i] > 0]
    pos_improve = [improvement[i] for i in range(len(improvement)) if improvement[i] > 0]
    neg_steps = [steps[i] for i in range(len(improvement)) if improvement[i] < 0]
    neg_improve = [improvement[i] for i in range(len(improvement)) if improvement[i] < 0]

    if pos_steps:
        ax.fill_between(pos_steps, 0, pos_improve, alpha=0.3, color='green', label='Class Better')
    if neg_steps:
        ax.fill_between(neg_steps, neg_improve, 0, alpha=0.3, color='red', label='Individual Better')

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Class Advantage (Class - Best)', fontsize=12)
    ax.set_title('Collective Advantage Over Time', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Collaborative Classroom: Proving Collective Intelligence on CIFAR-10',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('collaborative_results.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # Clear any cached models
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Run experiment
    classroom = run_experiment()

    # Plot results
    plot_results(classroom)

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)