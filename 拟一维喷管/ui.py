import sys
from PyQt5.QtWidgets import QWidget, QToolTip, QPushButton, QApplication, QMessageBox, QDesktopWidget, QMainWindow, \
    QAction, qApp,QPushButton, QHBoxLayout, QVBoxLayout,QGridLayout,QLabel,QLineEdit
from PyQt5.QtGui import QFont

from PyQt5.QtCore import QCoreApplication

import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


class MyFigure(FigureCanvas):
    def __init__(self, width=5, height=4, dpi=100):
        # 第一步：创建一个创建Figure
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        # 第二步：在父类中激活Figure窗口
        super(MyFigure, self).__init__(self.fig)  # 此句必不可少，否则不能显示图形
        # 第三步：创建一个子图，用于绘制图形用，111表示子图编号，如matlab的subplot(1,1,1)
        self.axes = self.fig.add_subplot(111)

class Example(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):

        # 这种静态的方法设置一个用于显示工具提示的字体。我们使用10px滑体字体。
        QToolTip.setFont(QFont('SansSerif', 10))

        # 显示状态栏
        self.setStatus('Ready')
        # 显示菜单
        self.initMenu()

        widget=QWidget()
        okButton = QPushButton("OK")
        cancelButton = QPushButton("Cancel")


        inputfile = QLabel('Input File')
        inputfileEdit = QLineEdit()

        grid = QGridLayout()
        grid.setSpacing(10)
        grid.addWidget(inputfile,1,0)
        grid.addWidget(inputfileEdit,1,1,1,3)

        F = MyFigure(width=3, height=2, dpi=100)
        t = np.arange(0.0, 10, 0.01)
        s = np.cos(2 * np.pi * t)
        F.axes.plot(t, s)  # 使用控件，在widget上作图
        F.fig.suptitle("cos")
        grid.addWidget(F,2,0,3,4)

        mpl_ntb = NavigationToolbar(F, self)
        grid.addWidget(mpl_ntb,6,0,1,4)

        widget.setLayout(grid)

        self.setCentralWidget(widget)

        # 控制窗口大小与位置
        self.resize(800, 450)

        self.center()

        # self.setGeometry(300, 300, 300, 200)
        self.setWindowTitle('CSM')
        self.show()


    # 控制窗口显示在屏幕中心的方法
    def center(self):
        # 获得窗口
        qr = self.frameGeometry()
        # 获得屏幕中心点
        cp = QDesktopWidget().availableGeometry().center()
        # 显示到屏幕中心
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def initMenu(self):
        # 显示菜单栏
        menubar = self.menuBar()
        # 添加菜单
        fileMenu = menubar.addMenu('&File')
        toolMenu = menubar.addMenu('&Tool')
        aboutMenu = menubar.addMenu('&About')

        # 在菜单中添加退出
        exitAction = QAction('&Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(QCoreApplication.instance().quit)
        fileMenu.addAction(exitAction)

    def setStatus(self,msg):
        self.statusBar().showMessage(msg)



if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())
