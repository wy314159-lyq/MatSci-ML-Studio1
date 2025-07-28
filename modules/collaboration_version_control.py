"""
Collaboration and Version Control Module for MatSci-ML Studio
协作和版本控制系统
"""

import os
import json
import datetime
import hashlib
import shutil
from typing import Dict, List, Optional, Any
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
                            QLabel, QPushButton, QTextEdit, QComboBox,
                            QLineEdit, QTableWidget, QTableWidgetItem,
                            QTabWidget, QFileDialog, QMessageBox, QInputDialog)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont

class ProjectManager:
    """项目管理器"""
    
    def __init__(self, base_path: str = "projects"):
        self.base_path = base_path
        self.ensure_project_directory()
    
    def ensure_project_directory(self):
        """确保项目目录存在"""
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)
    
    def create_project(self, project_name: str, description: str = "", author: str = "") -> str:
        """创建新项目"""
        project_id = self._generate_project_id(project_name)
        project_path = os.path.join(self.base_path, project_id)
        
        # 创建项目目录结构
        os.makedirs(project_path, exist_ok=True)
        os.makedirs(os.path.join(project_path, "data"), exist_ok=True)
        os.makedirs(os.path.join(project_path, "models"), exist_ok=True)
        os.makedirs(os.path.join(project_path, "results"), exist_ok=True)
        os.makedirs(os.path.join(project_path, "versions"), exist_ok=True)
        
        # 创建项目元数据
        metadata = {
            'project_id': project_id,
            'name': project_name,
            'description': description,
            'author': author,
            'created_at': datetime.datetime.now().isoformat(),
            'last_modified': datetime.datetime.now().isoformat(),
            'version': '1.0.0',
            'collaborators': [author] if author else [],
            'tags': [],
            'status': 'active'
        }
        
        self._save_project_metadata(project_path, metadata)
        return project_id
    
    def _generate_project_id(self, project_name: str) -> str:
        """生成项目ID"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        name_hash = hashlib.md5(project_name.encode()).hexdigest()[:8]
        return f"{project_name.replace(' ', '_')}_{timestamp}_{name_hash}"
    
    def _save_project_metadata(self, project_path: str, metadata: Dict):
        """保存项目元数据"""
        metadata_path = os.path.join(project_path, "project.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    def load_project_metadata(self, project_id: str) -> Dict:
        """加载项目元数据"""
        project_path = os.path.join(self.base_path, project_id)
        metadata_path = os.path.join(project_path, "project.json")
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def list_projects(self) -> List[Dict]:
        """列出所有项目"""
        projects = []
        
        if not os.path.exists(self.base_path):
            return projects
        
        for item in os.listdir(self.base_path):
            if os.path.isdir(os.path.join(self.base_path, item)):
                metadata = self.load_project_metadata(item)
                if metadata:
                    projects.append(metadata)
        
        return sorted(projects, key=lambda x: x.get('last_modified', ''), reverse=True)
    
    def delete_project(self, project_id: str) -> bool:
        """删除项目"""
        project_path = os.path.join(self.base_path, project_id)
        try:
            if os.path.exists(project_path):
                shutil.rmtree(project_path)
                return True
        except Exception as e:
            print(f"Failed to delete project: {e}")
        return False

class VersionControl:
    """版本控制系统"""
    
    def __init__(self, project_path: str):
        self.project_path = project_path
        self.versions_path = os.path.join(project_path, "versions")
        self.ensure_versions_directory()
    
    def ensure_versions_directory(self):
        """确保版本目录存在"""
        if not os.path.exists(self.versions_path):
            os.makedirs(self.versions_path)
    
    def create_snapshot(self, description: str = "", author: str = "") -> str:
        """创建快照"""
        timestamp = datetime.datetime.now()
        version_id = timestamp.strftime("%Y%m%d_%H%M%S")
        version_path = os.path.join(self.versions_path, version_id)
        
        os.makedirs(version_path, exist_ok=True)
        
        # 创建版本元数据
        version_metadata = {
            'version_id': version_id,
            'timestamp': timestamp.isoformat(),
            'author': author,
            'description': description,
            'files': []
        }
        
        # 复制项目文件
        files_copied = []
        
        for folder in ['data', 'models', 'results']:
            src_path = os.path.join(self.project_path, folder)
            if os.path.exists(src_path):
                dst_path = os.path.join(version_path, folder)
                shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                files_copied.extend(self._list_files_recursive(dst_path))
        
        version_metadata['files'] = files_copied
        
        # 保存版本元数据
        metadata_path = os.path.join(version_path, "version.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(version_metadata, f, indent=2, ensure_ascii=False)
        
        return version_id
    
    def _list_files_recursive(self, directory: str) -> List[str]:
        """递归列出目录中的所有文件"""
        files = []
        for root, dirs, filenames in os.walk(directory):
            for filename in filenames:
                file_path = os.path.join(root, filename)
                relative_path = os.path.relpath(file_path, directory)
                files.append(relative_path)
        return files
    
    def list_versions(self) -> List[Dict]:
        """列出所有版本"""
        versions = []
        
        if not os.path.exists(self.versions_path):
            return versions
        
        for item in os.listdir(self.versions_path):
            if os.path.isdir(os.path.join(self.versions_path, item)):
                version_metadata_path = os.path.join(self.versions_path, item, "version.json")
                if os.path.exists(version_metadata_path):
                    with open(version_metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                        versions.append(metadata)
        
        return sorted(versions, key=lambda x: x.get('timestamp', ''), reverse=True)

class CollaborationWidget(QWidget):
    """协作和版本控制主界面"""
    
    # 信号
    project_created = pyqtSignal(str)
    project_loaded = pyqtSignal(str)
    version_created = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.project_manager = ProjectManager()
        self.current_project_id = None
        self.version_control = None
        
        self.init_ui()
        self.load_projects_list()
    
    def init_ui(self):
        """初始化用户界面"""
        layout = QVBoxLayout(self)
        
        # 标题
        title = QLabel("🤝 Collaboration & Version Control")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("QLabel { color: #2196F3; margin: 10px; }")
        layout.addWidget(title)
        
        # 创建选项卡
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        
        # 1. 项目管理
        self.create_project_management_tab()
        
        # 2. 版本控制
        self.create_version_control_tab()
        
        # 状态栏
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("QLabel { color: #666666; margin: 5px; }")
        layout.addWidget(self.status_label)
    
    def create_project_management_tab(self):
        """创建项目管理标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 项目操作按钮
        buttons_layout = QHBoxLayout()
        
        self.new_project_btn = QPushButton("📁 New Project")
        self.new_project_btn.clicked.connect(self.create_new_project)
        buttons_layout.addWidget(self.new_project_btn)
        
        self.open_project_btn = QPushButton("📂 Open Project")
        self.open_project_btn.clicked.connect(self.open_project)
        buttons_layout.addWidget(self.open_project_btn)
        
        self.delete_project_btn = QPushButton("🗑️ Delete Project")
        self.delete_project_btn.clicked.connect(self.delete_project)
        buttons_layout.addWidget(self.delete_project_btn)
        
        self.refresh_btn = QPushButton("🔄 Refresh")
        self.refresh_btn.clicked.connect(self.load_projects_list)
        buttons_layout.addWidget(self.refresh_btn)
        
        layout.addLayout(buttons_layout)
        
        # Project list
        projects_group = QGroupBox("📋 Project List")
        projects_layout = QVBoxLayout(projects_group)
        
        self.projects_table = QTableWidget()
        self.projects_table.setColumnCount(6)
        self.projects_table.setHorizontalHeaderLabels([
            'Project Name', 'Author', 'Created', 'Last Modified', 'Version', 'Status'
        ])
        self.projects_table.setSelectionBehavior(QTableWidget.SelectRows)
        projects_layout.addWidget(self.projects_table)
        
        layout.addWidget(projects_group)
        
        # Project details
        details_group = QGroupBox("📄 Project Details")
        details_layout = QVBoxLayout(details_group)
        
        self.project_details_text = QTextEdit()
        self.project_details_text.setReadOnly(True)
        self.project_details_text.setMaximumHeight(150)
        details_layout.addWidget(self.project_details_text)
        
        layout.addWidget(details_group)
        
        self.tabs.addTab(tab, "Project Management")
        
        # 连接表格选择事件
        self.projects_table.selectionModel().selectionChanged.connect(self.on_project_selected)
    
    def create_version_control_tab(self):
        """创建版本控制标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Current project information
        current_project_group = QGroupBox("📁 Current Project")
        current_project_layout = QVBoxLayout(current_project_group)
        
        self.current_project_label = QLabel("No project selected")
        self.current_project_label.setFont(QFont("Arial", 12, QFont.Bold))
        current_project_layout.addWidget(self.current_project_label)
        
        layout.addWidget(current_project_group)
        
        # Version control buttons
        version_buttons_layout = QHBoxLayout()
        
        self.create_snapshot_btn = QPushButton("📸 Create Snapshot")
        self.create_snapshot_btn.clicked.connect(self.create_snapshot)
        self.create_snapshot_btn.setEnabled(False)
        version_buttons_layout.addWidget(self.create_snapshot_btn)
        
        self.export_project_btn = QPushButton("📦 Export Project")
        self.export_project_btn.clicked.connect(self.export_project)
        self.export_project_btn.setEnabled(False)
        version_buttons_layout.addWidget(self.export_project_btn)
        
        layout.addLayout(version_buttons_layout)
        
        # Version list
        versions_group = QGroupBox("🕐 Version History")
        versions_layout = QVBoxLayout(versions_group)
        
        self.versions_table = QTableWidget()
        self.versions_table.setColumnCount(4)
        self.versions_table.setHorizontalHeaderLabels([
            'Version ID', 'Created', 'Author', 'Description'
        ])
        self.versions_table.setSelectionBehavior(QTableWidget.SelectRows)
        versions_layout.addWidget(self.versions_table)
        
        layout.addWidget(versions_group)
        
        self.tabs.addTab(tab, "Version Control")
    
    def load_projects_list(self):
        """加载项目列表"""
        projects = self.project_manager.list_projects()
        
        self.projects_table.setRowCount(len(projects))
        
        for row, project in enumerate(projects):
            self.projects_table.setItem(row, 0, QTableWidgetItem(project.get('name', '')))
            self.projects_table.setItem(row, 1, QTableWidgetItem(project.get('author', '')))
            self.projects_table.setItem(row, 2, QTableWidgetItem(project.get('created_at', '')[:16]))
            self.projects_table.setItem(row, 3, QTableWidgetItem(project.get('last_modified', '')[:16]))
            self.projects_table.setItem(row, 4, QTableWidgetItem(project.get('version', '')))
            self.projects_table.setItem(row, 5, QTableWidgetItem(project.get('status', '')))
            
            # 保存项目ID
            self.projects_table.item(row, 0).setData(Qt.UserRole, project.get('project_id'))
        
        self.projects_table.resizeColumnsToContents()
    
    def on_project_selected(self):
        """项目选择事件"""
        selected_rows = self.projects_table.selectionModel().selectedRows()
        
        if selected_rows:
            row = selected_rows[0].row()
            project_id = self.projects_table.item(row, 0).data(Qt.UserRole)
            
            if project_id:
                metadata = self.project_manager.load_project_metadata(project_id)
                
                details_text = f"""Project ID: {metadata.get('project_id', '')}
Project name: {metadata.get('name', '')}
Description: {metadata.get('description', '')}
Author: {metadata.get('author', '')}
Created at: {metadata.get('created_at', '')}
Last modified: {metadata.get('last_modified', '')}
Version: {metadata.get('version', '')}
Collaborators: {', '.join(metadata.get('collaborators', []))}
Status: {metadata.get('status', '')}"""
                
                self.project_details_text.setText(details_text)
    
    def create_new_project(self):
        """创建新项目"""
        # 获取项目信息
        name, ok1 = QInputDialog.getText(self, "New Project", "Project name:")
        if not ok1 or not name.strip():
            return
        
        description, ok2 = QInputDialog.getText(self, "New Project", "Project description:")
        if not ok2:
            description = ""
        
        author, ok3 = QInputDialog.getText(self, "New Project", "Author:")
        if not ok3:
            author = ""
        
        # 创建项目
        try:
            project_id = self.project_manager.create_project(name.strip(), description, author)
            self.project_created.emit(project_id)
            self.load_projects_list()
            self.status_label.setText(f"Project '{name}' created successfully")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create project: {str(e)}")
    
    def open_project(self):
        """打开项目"""
        selected_rows = self.projects_table.selectionModel().selectedRows()
        
        if not selected_rows:
            QMessageBox.warning(self, "Warning", "Please select a project first")
            return
        
        row = selected_rows[0].row()
        project_id = self.projects_table.item(row, 0).data(Qt.UserRole)
        project_name = self.projects_table.item(row, 0).text()
        
        if project_id:
            self.current_project_id = project_id
            self.current_project_label.setText(f"Current project: {project_name}")
            
            # 初始化版本控制
            project_path = os.path.join(self.project_manager.base_path, project_id)
            self.version_control = VersionControl(project_path)
            
            # 启用版本控制按钮
            self.create_snapshot_btn.setEnabled(True)
            self.export_project_btn.setEnabled(True)
            
            # 加载版本列表
            self.load_versions_list()
            
            self.project_loaded.emit(project_id)
            self.status_label.setText(f"Project '{project_name}' opened")
    
    def delete_project(self):
        """删除项目"""
        selected_rows = self.projects_table.selectionModel().selectedRows()
        
        if not selected_rows:
            QMessageBox.warning(self, "Warning", "Please select a project first")
            return
        
        row = selected_rows[0].row()
        project_id = self.projects_table.item(row, 0).data(Qt.UserRole)
        project_name = self.projects_table.item(row, 0).text()
        
        # 确认删除
        reply = QMessageBox.question(
            self, "Confirm Delete", 
            f"Are you sure you want to delete project '{project_name}'?\nThis action cannot be undone!",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            if self.project_manager.delete_project(project_id):
                self.load_projects_list()
                self.status_label.setText(f"Project '{project_name}' deleted")
                
                # 如果删除的是当前项目，重置状态
                if self.current_project_id == project_id:
                    self.current_project_id = None
                    self.current_project_label.setText("No project selected")
                    self.create_snapshot_btn.setEnabled(False)
                    self.export_project_btn.setEnabled(False)
            else:
                QMessageBox.critical(self, "Error", "Failed to delete project")
    
    def load_versions_list(self):
        """加载版本列表"""
        if not self.version_control:
            return
        
        versions = self.version_control.list_versions()
        
        self.versions_table.setRowCount(len(versions))
        
        for row, version in enumerate(versions):
            self.versions_table.setItem(row, 0, QTableWidgetItem(version.get('version_id', '')))
            self.versions_table.setItem(row, 1, QTableWidgetItem(version.get('timestamp', '')[:16]))
            self.versions_table.setItem(row, 2, QTableWidgetItem(version.get('author', '')))
            self.versions_table.setItem(row, 3, QTableWidgetItem(version.get('description', '')))
        
        self.versions_table.resizeColumnsToContents()
    
    def create_snapshot(self):
        """创建快照"""
        if not self.version_control:
            QMessageBox.warning(self, "Warning", "Please select a project first")
            return
        
        # 获取快照信息
        description, ok1 = QInputDialog.getText(self, "Create Snapshot", "Snapshot description:")
        if not ok1:
            return
        
        author, ok2 = QInputDialog.getText(self, "Create Snapshot", "Author:")
        if not ok2:
            author = ""
        
        try:
            version_id = self.version_control.create_snapshot(description, author)
            self.load_versions_list()
            self.version_created.emit(version_id)
            self.status_label.setText(f"Snapshot {version_id} created successfully")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create snapshot: {str(e)}")
    
    def export_project(self):
        """导出项目包"""
        if not self.current_project_id:
            QMessageBox.warning(self, "Warning", "Please select a project first")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Project", f"{self.current_project_id}.zip", "Zip Files (*.zip)"
        )
        
        if file_path:
            try:
                # 创建项目包
                project_path = os.path.join(self.project_manager.base_path, self.current_project_id)
                shutil.make_archive(file_path[:-4], 'zip', project_path)
                
                self.status_label.setText(f"Project package exported to: {file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export project: {str(e)}") 