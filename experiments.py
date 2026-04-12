from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch
import collaborative_classroom as CollaborativeClassroom
import numpy as np


class Experiments:

    @staticmethod
    def create_data_stream(batch_size=32, val_batch_size=100):
        # CIFAR-10 specific transforms with data augmentation
        # Data augmentation helps prevent overfitting on the smaller CIFAR dataset

        # Training transforms with augmentation
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),  # Mirror images (valid for CIFAR)
            transforms.RandomCrop(32, padding=4),  # Slight random crops
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],  # CIFAR-10 mean per channel
                                 std=[0.2470, 0.2435, 0.2616])  # CIFAR-10 std per channel
        ])

        # Validation/Test transforms (no augmentation, just normalization)
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                 std=[0.2470, 0.2435, 0.2616])
        ])

        print("Loading CIFAR-10 dataset (32x32 color images, 10 classes)...")
        print("Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck")

        # Load CIFAR-10 datasets
        full_train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=train_transform)

        # Split: 80% training, 20% validation
        train_size = int(0.8 * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_train_dataset, [train_size, val_size]
        )

        test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=test_transform)

        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,  # Faster loading
            pin_memory=True  # Faster GPU transfer if available
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1000,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

        print(f"Training samples: {train_size:,}")
        print(f"Validation samples: {val_size:,}")
        print(f"Test samples: {len(test_dataset):,}")
        print(f"Image shape: 3x32x32 (RGB)")

        return train_loader, val_loader, test_loader

    @staticmethod
    def run_experiment():
        print("=" * 70)
        print("COLLABORATIVE CLASSROOM EXPERIMENT - CIFAR-10")
        print("Testing: Can collective intelligence surpass individual performance?")
        print("=" * 70)

        # CIFAR-10 requires more students for diversity
        classroom = CollaborativeClassroom.CollaborativeClassroom(
            num_students=12,  # More students for CIFAR's complexity
            moving_avg_window=100,
            eval_interval_for_best_student=200,
            val_batch_size=100
        )

        train_loader, val_loader, test_loader = Experiments.create_data_stream(
            batch_size=64,  # Smaller batch size for CIFAR
            val_batch_size=100
        )
        classroom.setup_validation_batches(val_loader, num_batches=5)

        steps = 10000  # More steps for CIFAR (harder problem)
        class_evaluation_every = 250

        print(f"\nStarting training for {steps} steps...")
        print("This will take longer due to CNN architecture and dataset complexity")
        print("=" * 70)

        data_iter = iter(train_loader)
        for step in range(steps):
            try:
                x, y_true = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                x, y_true = next(data_iter)

            classroom.learn_step(x, y_true, step)

            if step % class_evaluation_every == 0 or step == steps - 1:
                teacher_acc, best_acc, class_acc = classroom.evaluate_class(test_loader, step)

                advantage = class_acc - best_acc
                advantage_symbol = "▲" if advantage > 0 else "▼" if advantage < 0 else "●"

                print(f"Step {step:5d} | Teacher: {teacher_acc:.3f} | "
                      f"Best: {best_acc:.3f} | Class: {class_acc:.3f} | "
                      f"Advantage: {advantage_symbol}{advantage:+.4f} | "
                      f"Phase: {classroom.phase[:5]} | Best ID: {classroom.best_student_idx}")

        print("\n" + "=" * 70)
        print("EXPERIMENT RESULTS")
        print("=" * 70)
        print(f"Number of students: {classroom.num_students}")
        print(f"Teacher calls used: {1000 - classroom.teacher.calls_remaining}/1000")

        final_class_acc = classroom.history['class_avg_accuracy'][-1]
        final_best_acc = classroom.history['best_student_accuracy'][-1]
        final_advantage = final_class_acc - final_best_acc

        if classroom.history['class_surpassed_best']:
            print(f"\n✅ SUCCESS: Collective intelligence emerged on CIFAR-10!")
            print(f"   Class surpassed best student at step {classroom.history['step_when_surpassed']}")
            print(f"   Final Class Accuracy: {final_class_acc:.4f} ({final_class_acc * 100:.1f}%)")
            print(f"   Final Best Student Accuracy: {final_best_acc:.4f} ({final_best_acc * 100:.1f}%)")
            print(f"   Improvement: +{final_advantage:.4f} (+{final_advantage * 100:.1f}%)")
            print(f"\n   This proves that {classroom.num_students} students working together")
            print(f"   outperform even their best individual member!")
        else:
            print(f"\n⚠️  Collective advantage not yet achieved on CIFAR-10")
            print(f"   Final Class: {final_class_acc:.4f}, Best: {final_best_acc:.4f}")
            print(f"   Advantage: {final_advantage:+.4f}")
            print(f"\n   Suggestion: Increase num_students or run for more steps")

        return classroom

    @staticmethod
    def plot_results(classroom):
        plt.figure(figsize=(16, 12))

        steps = range(0, len(classroom.history['teacher_accuracy']) * 250, 250)

        # Plot 1: Main accuracy comparison
        plt.subplot(2, 2, 1)
        plt.plot(steps, classroom.history['teacher_accuracy'], 'g-',
                 label='Teacher (Oracle)', alpha=0.5, linewidth=1)
        plt.plot(steps, classroom.history['best_student_accuracy'], 'r-',
                 label='Best Student', linewidth=2)
        plt.plot(steps, classroom.history['class_avg_accuracy'], 'b-',
                 label='Class Collective (Majority Vote)', linewidth=2.5)

        if classroom.history['class_surpassed_best']:
            surpass_step = classroom.history['step_when_surpassed']
            plt.axvline(x=surpass_step, color='purple', linestyle='--', linewidth=2,
                        label=f'Collective Surpasses Individual (step {surpass_step})')

            surpass_idx = min(int(surpass_step / 250), len(steps) - 1)
            plt.fill_between(steps[surpass_idx:],
                             classroom.history['best_student_accuracy'][surpass_idx:],
                             classroom.history['class_avg_accuracy'][surpass_idx:],
                             alpha=0.3, color='blue', label='Collective Advantage Region')

        plt.xlabel('Training Step', fontsize=11)
        plt.ylabel('Accuracy', fontsize=11)
        plt.title('CIFAR-10: Collective Intelligence vs Individual Performance', fontsize=12)
        plt.legend(loc='lower right', fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)

        # Plot 2: Student performance distribution
        plt.subplot(2, 2, 2)
        if classroom.history['individual_accuracies']:
            final_data = classroom.history['individual_accuracies'][-1]
            final_accs = final_data['accuracies']
            best_idx = final_data['best_idx']

            colors = ['red' if i == best_idx else 'steelblue' for i in range(len(final_accs))]
            bars = plt.bar(range(len(final_accs)), final_accs, color=colors, alpha=0.7)
            plt.axhline(y=classroom.history['class_avg_accuracy'][-1], color='blue',
                        linestyle='--', linewidth=2,
                        label=f'Class Avg: {classroom.history["class_avg_accuracy"][-1]:.3f}')
            plt.xlabel('Student ID', fontsize=11)
            plt.ylabel('Final Accuracy', fontsize=11)
            plt.title('Student Diversity on CIFAR-10', fontsize=12)
            plt.legend()

            for bar, acc in zip(bars, final_accs):
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                         f'{acc:.3f}', ha='center', va='bottom', fontsize=8)

        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)

        # Plot 3: Learning phases
        plt.subplot(2, 2, 3)
        teacher_steps = 2000
        plt.fill_between(steps, 0, 1,
                         where=[s < teacher_steps for s in steps],
                         alpha=0.3, color='green', label='Teacher Phase (0-2000 steps)')
        plt.fill_between(steps, 0, 1,
                         where=[s >= teacher_steps for s in steps],
                         alpha=0.3, color='orange', label='Peer Learning Phase')

        plt.plot(steps, classroom.history['class_avg_accuracy'], 'b-',
                 label='Class Accuracy', linewidth=2)
        plt.plot(steps, classroom.history['best_student_accuracy'], 'r-',
                 label='Best Student', linewidth=1.5)

        plt.xlabel('Training Step', fontsize=11)
        plt.title('Learning Phases on CIFAR-10', fontsize=12)
        plt.yticks([])
        plt.ylim(0, 1)
        plt.legend(loc='lower right', fontsize=9)

        # Plot 4: Final performance comparison
        plt.subplot(2, 2, 4)
        final_class = classroom.history['class_avg_accuracy'][-1]
        final_best = classroom.history['best_student_accuracy'][-1]
        improvement = final_class - final_best

        categories = ['Best Student\n(Individual Expert)',
                      'Class Collective\n(12 Students Voting)']
        final_accuracies = [final_best, final_class]
        colors = ['#ff6b6b', '#4ecdc4']
        bars = plt.bar(categories, final_accuracies, color=colors, alpha=0.8,
                       edgecolor='black', linewidth=1.5)

        if improvement > 0:
            plt.annotate(f'▲ +{improvement:.3f}\n({improvement * 100:.1f}% better)',
                         xy=(1, final_class), xytext=(1.2, final_class - 0.1),
                         arrowprops=dict(arrowstyle='->', color='green', lw=2),
                         fontsize=10, ha='left', color='green', fontweight='bold')

        plt.ylabel('Accuracy', fontsize=11)
        title = f'CIFAR-10: Collective {"BEATS" if improvement > 0 else "TIES"} Individual'
        plt.title(title, fontsize=12, color='green' if improvement > 0 else 'orange')

        for bar, acc in zip(bars, final_accuracies):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{acc:.3f}\n({acc * 100:.1f}%)', ha='center', va='bottom',
                     fontsize=10, fontweight='bold')

        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3, axis='y')

        plt.suptitle('Collaborative Classroom on CIFAR-10: Proving Collective Intelligence',
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig('collaborative_classroom_cifar10_results.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Print detailed analysis
        print("\n" + "=" * 70)
        print("FINAL ANALYSIS - CIFAR-10 EXPERIMENT")
        print("=" * 70)
        if classroom.history['class_surpassed_best']:
            print("✅ EXPERIMENT SUCCESSFUL: Collective Intelligence Demonstrated on CIFAR-10")
            print(f"\n   Key Findings:")

            print(f"   • Collective improvement: +{improvement * 100:.1f}%")
            print(f"\n   Why this matters for CIFAR-10:")
            print(f"   • CIFAR-10 is significantly harder than MNIST (color, complex objects)")
            print(f"   • Students make different errors (e.g., confusing cats with dogs)")
            print(f"   • Majority voting cancels out individual mistakes")
            print(f"   • This proves emergent intelligence scales to complex problems")
        else:
            print("⚠️  Run again or increase training steps - CIFAR-10 is challenging!")


if __name__ == "__main__":
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        print("Training will be faster!")
    else:
        print("GPU not available - training will use CPU (slower but works)")

    classroom = Experiments.run_experiment()
    Experiments.plot_results(classroom)