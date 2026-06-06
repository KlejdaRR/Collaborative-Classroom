import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
from Models.HamiltonianStudent import HamiltonianStudent
from Models.SimpleTeacher import SimpleTeacher


class HamiltonianCollaborativeClassroom:
    """
    Collaborative classroom with Hamiltonian dynamics.

    The costate regularization prevents individual students from
    dominating, maintaining collective diversity.
    """

    def __init__(self, num_students=8, device='cuda',
                 costate_decay=0.999, hamiltonian_dt=0.01):
        self.num_students = num_students
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.costate_decay = costate_decay
        self.hamiltonian_dt = hamiltonian_dt

        print(f"Hamiltonian Classroom using device: {self.device}")
        print(f"Costate decay: {costate_decay}, Hamiltonian dt: {hamiltonian_dt}")

        # Create Hamiltonian students
        self.students = []
        for i in range(num_students):
            student = HamiltonianStudent(
                num_classes=10,
                costate_decay=costate_decay,
                temporal_horizon=50
            )
            student = student.to_device(self.device)
            self.students.append(student)

        # Add diversity (important for collective intelligence)
        with torch.no_grad():
            for i, student in enumerate(self.students):
                if i > 0:
                    for param in student.parameters():
                        param.add_(torch.randn_like(param) * 0.1)

        # Teacher (simple, not Hamiltonian)
        self.teacher = SimpleTeacher(total_calls=2000)

        # Tracking
        self.best_student_idx = 0
        self.phase = "teacher_phase"

        # History for plotting
        self.history = {
            'teacher_acc': [],
            'best_acc': [],
            'class_acc': [],
            'steps': [],
            'costate_norms': []
        }

        # Validation data
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

        for student in self.students:
            student.eval()

        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(self.device)
                target = target.to(self.device)

                if self.teacher.get_remaining() > 0:
                    teacher_pred = self.students[self.best_student_idx](data)
                    teacher_pred = teacher_pred.argmax(dim=1)
                    teacher_correct += (teacher_pred == target).sum().item()

                for i, student in enumerate(self.students):
                    pred = student(data).argmax(dim=1)
                    individual_correct[i] += (pred == target).sum().item()

                best_pred = self.students[self.best_student_idx](data).argmax(dim=1)
                best_correct += (best_pred == target).sum().item()

                all_preds = torch.stack([s(data).argmax(dim=1) for s in self.students])
                class_pred = torch.mode(all_preds, dim=0)[0]
                class_correct += (class_pred == target).sum().item()

                total += len(data)

            # Get costate norm from best student
            metrics = self.students[self.best_student_idx].get_metrics()
            avg_costate = metrics['avg_costate_norm']

        for student in self.students:
            student.train()

        teacher_acc = teacher_correct / total if self.teacher.get_remaining() > 0 else 0
        best_acc = best_correct / total
        class_acc = class_correct / total

        self.history['teacher_acc'].append(teacher_acc)
        self.history['best_acc'].append(best_acc)
        self.history['class_acc'].append(class_acc)
        self.history['steps'].append(step)
        self.history['costate_norms'].append((step, avg_costate))

        return teacher_acc, best_acc, class_acc, individual_correct

    def learn_step(self, x, y_true, step):
        """One learning step with Hamiltonian dynamics"""
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

        teacher_answer = self.teacher.teach(x, y_true)

        # Teacher phase
        if self.phase == "teacher_phase" and teacher_answer is not None:
            for student in self.students:
                student.hamiltonian_step(x, teacher_answer,
                                         learning_rate=0.001,
                                         dt=self.hamiltonian_dt)

        # Peer learning phase
        elif self.phase == "peer_phase":
            with torch.no_grad():
                best_output = self.students[self.best_student_idx](x)
                soft_targets = F.softmax(best_output / 2.0, dim=1)

            for i, student in enumerate(self.students):
                if i == self.best_student_idx:
                    if teacher_answer is not None:
                        student.hamiltonian_step(x, teacher_answer,
                                                 learning_rate=0.001,
                                                 dt=self.hamiltonian_dt)
                    continue

                pseudo_target = soft_targets.argmax(dim=1)
                student.hamiltonian_step(x, pseudo_target,
                                         learning_rate=0.0005,
                                         dt=self.hamiltonian_dt * 0.5)