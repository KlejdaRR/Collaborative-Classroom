import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
from Models.SimpleStudent import SimpleStudent
from Models.SimpleTeacher import SimpleTeacher


class SimpleCollaborativeClassroom:
    """
    Simplified collaborative classroom that actually learns
    """

    def __init__(self, num_students=8, device='cuda'):
        self.num_students = num_students
        self.device = device if torch.cuda.is_available() else 'cpu'

        print(f"Classroom using device: {self.device}")

        # Create students and move to device
        self.students = []
        for i in range(num_students):
            student = SimpleStudent()
            student = student.to(self.device)
            self.students.append(student)

        # Add diversity
        with torch.no_grad():
            for i, student in enumerate(self.students):
                if i > 0:
                    for param in student.parameters():
                        param.add_(torch.randn_like(param) * 0.1)

        # Optimizers
        self.optimizers = [optim.Adam(student.parameters(), lr=0.001)
                           for student in self.students]

        # Teacher
        self.teacher = SimpleTeacher(total_calls=2000)

        # Tracking
        self.best_student_idx = 0
        self.phase = "teacher_phase"

        # History
        self.history = {
            'teacher_acc': [],
            'best_acc': [],
            'class_acc': [],
            'steps': []
        }

        # Validation data (will be moved to device)
        self.val_data = None
        self.val_labels = None

    def set_validation_data(self, val_loader, num_batches=5):
        """Store validation data for best student selection"""
        val_data_list = []
        val_labels_list = []

        for i, (data, labels) in enumerate(val_loader):
            if i >= num_batches:
                break
            val_data_list.append(data)
            val_labels_list.append(labels)

        self.val_data = torch.cat(val_data_list, dim=0).to(self.device)
        self.val_labels = torch.cat(val_labels_list, dim=0).to(self.device)
        print(f"Validation set: {len(self.val_data)} samples")

    def evaluate_student(self, student):
        """Evaluate a single student on validation data"""
        if self.val_data is None:
            return 0

        student.eval()
        with torch.no_grad():
            output = student(self.val_data)
            pred = output.argmax(dim=1)
            correct = (pred == self.val_labels).sum().item()
            return correct / len(self.val_data)

    def find_best_student(self):
        """Find the best performing student"""
        accuracies = [self.evaluate_student(s) for s in self.students]
        return np.argmax(accuracies)

    def evaluate_class(self, test_loader, step):
        """Evaluate the entire classroom"""
        teacher_correct = 0
        best_correct = 0
        class_correct = 0
        total = 0

        individual_correct = [0] * self.num_students

        # Set all students to eval mode
        for student in self.students:
            student.eval()

        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(self.device)
                target = target.to(self.device)

                # Teacher (if still active)
                if self.teacher.get_remaining() > 0:
                    teacher_pred = self.students[self.best_student_idx](data)
                    teacher_pred = teacher_pred.argmax(dim=1)
                    teacher_correct += (teacher_pred == target).sum().item()

                # Individual students
                for i, student in enumerate(self.students):
                    pred = student(data).argmax(dim=1)
                    individual_correct[i] += (pred == target).sum().item()

                # Best student
                best_pred = self.students[self.best_student_idx](data).argmax(dim=1)
                best_correct += (best_pred == target).sum().item()

                # Class collective (majority vote)
                all_preds = torch.stack([s(data).argmax(dim=1) for s in self.students])
                class_pred = torch.mode(all_preds, dim=0)[0]
                class_correct += (class_pred == target).sum().item()

                total += len(data)

        # Set back to train mode
        for student in self.students:
            student.train()

        teacher_acc = teacher_correct / total if self.teacher.get_remaining() > 0 else 0
        best_acc = best_correct / total
        class_acc = class_correct / total

        self.history['teacher_acc'].append(teacher_acc)
        self.history['best_acc'].append(best_acc)
        self.history['class_acc'].append(class_acc)
        self.history['steps'].append(step)

        return teacher_acc, best_acc, class_acc, individual_correct

    def learn_step(self, x, y_true, step):
        """One learning step"""

        # Ensure data is on correct device
        x = x.to(self.device)
        y_true = y_true.to(self.device)

        # Phase management
        if self.teacher.get_remaining() > 0 and step < 3000:
            self.phase = "teacher_phase"
        else:
            self.phase = "peer_phase"

        # Update best student periodically
        if step % 500 == 0 and step > 0:
            old_best = self.best_student_idx
            self.best_student_idx = self.find_best_student()
            if old_best != self.best_student_idx:
                print(f"  Step {step}: Best student {old_best} → {self.best_student_idx}")

        # Get teacher answer
        teacher_answer = self.teacher.teach(x, y_true)

        # Teacher phase
        if self.phase == "teacher_phase" and teacher_answer is not None:
            for student, optimizer in zip(self.students, self.optimizers):
                optimizer.zero_grad()
                output = student(x)
                loss = F.nll_loss(output, teacher_answer)
                loss.backward()
                optimizer.step()

        # Peer learning phase
        elif self.phase == "peer_phase":
            with torch.no_grad():
                best_output = self.students[self.best_student_idx](x)
                soft_targets = F.softmax(best_output / 2.0, dim=1)

            for i, (student, optimizer) in enumerate(zip(self.students, self.optimizers)):
                if i == self.best_student_idx:
                    # Best student continues learning from teacher if available
                    if teacher_answer is not None:
                        optimizer.zero_grad()
                        output = student(x)
                        loss = F.nll_loss(output, teacher_answer)
                        loss.backward()
                        optimizer.step()
                    continue

                optimizer.zero_grad()
                student_output = student(x)

                # Distillation loss
                distill_loss = F.kl_div(
                    F.log_softmax(student_output / 2.0, dim=1),
                    soft_targets,
                    reduction='batchmean'
                )

                distill_loss.backward()
                optimizer.step()