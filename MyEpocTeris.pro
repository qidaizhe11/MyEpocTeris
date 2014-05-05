# Add more folders to ship with the application, here
folder_01.source = qml/MyEpocTeris
folder_01.target = qml
DEPLOYMENTFOLDERS = folder_01

# Additional import path used to resolve QML modules in Creator's code model
QML_IMPORT_PATH =

# The .cpp file which was generated for your project. Feel free to hack it.
SOURCES += main.cpp

# Installation path
# target.path =

# Please do not modify the following two lines. Required for deployment.
include(qtquick2applicationviewer/qtquick2applicationviewer.pri)
qtcAddDeployment()

OTHER_FILES += \
    EmotivClassify/1_arma_v2.mat

HEADERS += \
    EmotivClassify/classify.h \
    EmotivClassify/corr2.h \
    EmotivClassify/FDA_TEST.h \
    EmotivClassify/FDA_TRAIN.h \
    EmotivClassify/filter.h \
    EmotivClassify/gotBlockFlags.h \
    EmotivClassify/gotPlv.h \
    EmotivClassify/my_csp.h \
    EmotivClassify/my_unwrap.h
