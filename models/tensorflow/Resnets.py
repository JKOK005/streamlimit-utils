class Resnet():
    def resnet50():
        '''
        Returns resnet50 model in tensorflow, input size is 224, 224, 3
        '''
        import tensorflow as tf
        return tf.keras.applications.ResNet50(weights=None)

    def resnet152():
        '''
        Returns resnet152 model in tensorflow, input size is 224, 224, 3
        '''
        import tensorflow as tf
        return tf.keras.applications.ResNet152(weights=None)