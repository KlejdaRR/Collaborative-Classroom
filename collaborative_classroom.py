import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
from Models.Teacher import Teacher
from Models.Student import Student


class CollaborativeClassroom:
    def __init__(self, num_students=8, moving_avg_window=100, eval_interval_for_best_student=200, val_batch_size=100):
        self.num_students = num_students
        self.moving_avg_window = moving_avg_window
        self.eval_interval_for_best_student = eval_interval_for_best_student
        self.val_batch_size = val_batch_size

        self.students = [Student() for _ in range(num_students)]

        # MODIFICATION 2: Add diversity to student initializations
        # This ensures students start with different perspectives and make different errors
        with torch.no_grad():
            for i, student in enumerate(self.students):
                if i > 0:  # Keep first student as baseline, diversify others
                    for param in student.parameters():
                        # Add 5% random noise to break symmetry
                        param.add_(torch.randn_like(param) * 0.05)

        self.optimizers = [optim.Adam(student.parameters(), lr=0.001) for student in self.students]

        self.teacher = Teacher()
        self.loss_memory = [deque(maxlen=moving_avg_window) for _ in range(num_students)]

        self.best_student_idx = 0
        self.phase = "teacher_phase"

        self.validation_batches = []

        self.history = {
            'teacher_accuracy': [],
            'best_student_accuracy': [],
            'class_avg_accuracy': [],
            'class_surpassed_best': False,
            'step_when_surpassed': None,
            'individual_accuracies': []  # Track all students for analysis
        }

    def setup_validation_batches(self, val_loader, num_batches=5):
        self.validation_batches = []
        val_iter = iter(val_loader)

        for _ in range(num_batches):
            try:
                x_val, y_val = next(val_iter)
                self.validation_batches.append((x_val, y_val))
            except StopIteration:
                break

        print(f"Created {len(self.validation_batches)} validation batches for best student selection")

    def evaluate_student_on_validation(self, student_idx):
        student = self.students[student_idx]
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for x_val, y_val in self.validation_batches:
                output = student.forward(x_val)
                pred = output.argmax(dim=1, keepdim=True)
                comparison = pred.eq(y_val.view_as(pred))
                correct = comparison.sum().item()
                total_correct += correct
                total_samples += len(x_val)

        return total_correct / total_samples if total_samples > 0 else 0

    def find_best_student(self):
        if not self.validation_batches:
            avg_losses = [self.get_moving_average_loss(i) for i in range(self.num_students)]
            return np.argmin(avg_losses)

        val_accuracies = []
        for i in range(self.num_students):
            accuracy = self.evaluate_student_on_validation(i)
            val_accuracies.append(accuracy)

        return np.argmax(val_accuracies)

    def get_moving_average_loss(self, student_idx):
        if len(self.loss_memory[student_idx]) == 0:
            return float('inf')
        return np.mean(list(self.loss_memory[student_idx]))

    def evaluate_class(self, test_loader, step):
        teacher_correct = 0
        best_student_correct = 0
        class_total_correct = 0
        total_samples = 0

        # Track individual student performances
        individual_correct = [0] * self.num_students

        with torch.no_grad():
            for data, target in test_loader:
                if self.teacher.calls_remaining > 0:
                    teacher_pred = self.students[self.best_student_idx].forward(data)
                    teacher_pred = teacher_pred.argmax(dim=1, keepdim=True)
                    teacher_correct += teacher_pred.eq(target.view_as(teacher_pred)).sum().item()

                # Individual student accuracies
                for i, student in enumerate(self.students):
                    pred = student.forward(data)
                    pred = pred.argmax(dim=1, keepdim=True)
                    individual_correct[i] += pred.eq(target.view_as(pred)).sum().item()

                # Best Student Accuracy
                best_pred = self.students[self.best_student_idx].forward(data)
                best_pred = best_pred.argmax(dim=1, keepdim=True)
                best_student_correct += best_pred.eq(target.view_as(best_pred)).sum().item()

                # Class Collective Accuracy (Majority Voting)
                class_predictions = []
                for student in self.students:
                    pred = student.forward(data)
                    class_predictions.append(pred.argmax(dim=1, keepdim=True))

                class_votes = torch.stack(class_predictions)
                class_final_pred = torch.mode(class_votes, dim=0)[0]
                class_total_correct += class_final_pred.eq(target.view_as(class_final_pred)).sum().item()

                total_samples += len(data)

        teacher_acc = teacher_correct / total_samples if teacher_correct > 0 else 0
        best_acc = best_student_correct / total_samples
        class_acc = class_total_correct / total_samples

        # Store individual accuracies
        individual_accs = [correct / total_samples for correct in individual_correct]
        self.history['individual_accuracies'].append({
            'step': step,
            'accuracies': individual_accs,
            'best_idx': self.best_student_idx
        })

        self.history['teacher_accuracy'].append(teacher_acc)
        self.history['best_student_accuracy'].append(best_acc)
        self.history['class_avg_accuracy'].append(class_acc)

        # MODIFICATION 3: Lowered threshold from 500 to 200 for earlier breakthrough detection
        if (not self.history['class_surpassed_best'] and
                class_acc > best_acc and
                step > 200):
            self.history['class_surpassed_best'] = True
            self.history['step_when_surpassed'] = step
            margin = class_acc - best_acc
            print(f"\n🎉 BREAKTHROUGH! Class surpassed best student at step {step}!")
            print(f"   Class Accuracy: {class_acc:.4f} | Best Student: {best_acc:.4f} | Margin: +{margin:.4f}")

        return teacher_acc, best_acc, class_acc

    def learn_step(self, x, y_true, step):
        if self.teacher.calls_remaining > 0 and step < 2000:
            self.phase = "teacher_phase"
        else:
            self.phase = "peer_phase"

        if step % self.eval_interval_for_best_student == 0:
            old_best = self.best_student_idx
            self.best_student_idx = self.find_best_student()
            if old_best != self.best_student_idx and step > 0:
                print(f"  📊 Step {step}: Best student changed from {old_best} → {self.best_student_idx}")

        teacher_response = self.teacher.teach(x, y_true)

        if self.phase == "teacher_phase":
            if teacher_response is not None:
                for i, (student, optimizer) in enumerate(zip(self.students, self.optimizers)):
                    optimizer.zero_grad()
                    output = student.forward(x)
                    loss = F.nll_loss(output, teacher_response)
                    loss.backward()
                    optimizer.step()
                    self.loss_memory[i].append(loss.item())
            else:
                self.phase = "peer_phase"

        if self.phase == "peer_phase":
            with torch.no_grad():
                best_student_output = self.students[self.best_student_idx](x)
                soft_targets = F.softmax(best_student_output / 2.0, dim=1)

            for i, (student, optimizer) in enumerate(zip(self.students, self.optimizers)):
                if i == self.best_student_idx:
                    if teacher_response is not None:
                        optimizer.zero_grad()
                        output = student.forward(x)
                        loss = F.nll_loss(output, teacher_response)
                        loss.backward()
                        optimizer.step()
                        self.loss_memory[i].append(loss.item())
                    continue

                optimizer.zero_grad()
                student_output = student.forward(x)

                distillation_loss = F.kl_div(
                    F.log_softmax(student_output / 2.0, dim=1),
                    soft_targets,
                    reduction='batchmean'
                )

                if teacher_response is not None:
                    classification_loss = F.nll_loss(student_output, teacher_response)
                    total_loss = 0.7 * distillation_loss + 0.3 * classification_loss
                else:
                    total_loss = distillation_loss

                total_loss.backward()
                optimizer.step()
                self.loss_memory[i].append(total_loss.item())