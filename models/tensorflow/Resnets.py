class Resnet():
    def resnet50(classes=10):
        '''
        Returns resnet50 model in tensorflow, input size is 224, 224, 3
        '''
        import tensorflow as tf
        return tf.keras.applications.ResNet50(weights=None, classes=classes)

    def resnet152(classes=10):
        '''
        Returns resnet152 model in tensorflow, input size is 224, 224, 3
        '''
        import tensorflow as tf
        return tf.keras.applications.ResNet152(weights=None, classes=classes)