import tensorflow as tf
import numpy as np
import torchfile

# VGG19 mean values
_RGB_MEANS = np.array([123.68, 116.78, 103.94])
_BGR_MEANS = np.array([103.94, 116.78, 123.68])

def any_to_uint8_scale(image):
    '''
    Scales a numpy array to [0, 255] and converts it to uint8 
    '''
    float_image = image.astype(np.float32)
    imax = float_image.max()
    imin = float_image.min()
    diff = abs(imax - imin)
    normalized_image = (float_image - imin) / diff
    return (normalized_image * 255).astype(np.uint8)

def any_to_uint8_clip(image):
    '''
    Clips a numpy array to [0, 255] and converts it to uint8
    '''
    image = np.clip(image, 0, 255)
    return image.astype(np.uint8)

def graph_from_t7(net, graph, t7_file):
    '''
    Loads a Torch network from a saved .t7 file into Tensorflow.
    
    :param net Input to Torch network
    :param graph Tensorflow graph that the network should be created as part of
    :param t7 Path to t7 file to use
    '''
    layers = []
    t7 = torchfile.load(t7_file,force_8bytes_long=True)
    
    with graph.as_default():
        for module in t7.modules:
            if module._typename == b'nn.SpatialReflectionPadding':
                left = module.pad_l
                right = module.pad_r
                top = module.pad_t
                bottom = module.pad_b
                net = tf.pad(net, [[0,0], [top, bottom], [left, right], [0,0]], 'REFLECT')
                layers.append(net)
            elif module._typename == b'nn.SpatialConvolution':
                weight = module.weight.transpose([2,3,1,0])
                bias = module.bias
                strides = [1, module.dH, module.dW, 1]  # Assumes 'NHWC'
                net = tf.nn.conv2d(net, weight, strides, padding='VALID')
                net = tf.nn.bias_add(net, bias)
                layers.append(net)
            elif module._typename == b'nn.ReLU':
                net = tf.nn.relu(net)
                layers.append(net)
            elif module._typename == b'nn.SpatialUpSamplingNearest':
                d = tf.shape(net)
                size = [d[1] * module.scale_factor, d[2] * module.scale_factor]
                net = tf.image.resize_nearest_neighbor(net, size)
                layers.append(net)
            elif module._typename == b'nn.SpatialMaxPooling':
                net = tf.nn.max_pool(net, ksize=[1, module.kH, module.kW, 1], strides=[1, module.dH, module.dW, 1],
                                   padding='VALID', name = str(module.name, 'utf-8'))
                layers.append(net)
            else:
                raise NotImplementedError(module._typename)
        
        return net, layers

def _offset_image(image, means):
    image = tf.to_float(image)
    channels = tf.split(axis=3, num_or_size_splits=3, value=image)
    for i in range(3):
        channels[i] += means[i]
    image = tf.concat(axis=3, values=channels)
    return image
    
def preprocess_image(image, size=None):
    
    # Nets like VGG trained on images imported from OpenCV are
    # in BGR order, so we need to flip the channels on the incoming image.
    # Remove this part if not needed, but for now we assume inputs
    # are in RGB.
    # See https://github.com/jcjohnson/neural-style/issues/207 for
    # further discussion
    image = tf.reverse(image, axis=[-1])
    
    
    image = _offset_image(image, _BGR_MEANS)
    if size is not None:
        image = tf.image.resize_images(image, size)
    return image

def postprocess_image(image, size=None):
    image = _offset_image(image, -1*_BGR_MEANS)

    # Flip back to RGB
    image = tf.reverse(image, axis=[-1])
    return image
    
def image_from_file(graph, placeholder_name, size=None):
    with graph.as_default():
        filename = tf.placeholder(tf.string, name=placeholder_name)
        image = tf.image.decode_jpeg(tf.read_file(filename))
        image = tf.expand_dims(image, 0)
        image = preprocess_image(image, size)
        return image, filename

def AdaIN(content_features, style_features, alpha):
    '''
    Normalizes the `content_features` with scaling and offset from `style_features`.
    See "5. Adaptive Instance Normalization" in https://arxiv.org/abs/1703.06868 for details.
    '''
    style_mean, style_variance = tf.nn.moments(style_features, [1,2], keep_dims=True)
    content_mean, content_variance = tf.nn.moments(content_features, [1,2], keep_dims=True)
    epsilon = 1e-5
    normalized_content_features = tf.nn.batch_normalization(content_features, content_mean,
                                                            content_variance, style_mean, 
                                                            tf.sqrt(style_variance), epsilon)
    normalized_content_features = alpha * normalized_content_features + (1 - alpha) * content_features
    return normalized_content_features
    
    
def stylize(content, style, alpha, vgg_t7_file, decode_t7_file, resize=[512,512]):
    '''
    :param content Filename for the content image    
    :param style Filename for the style image
    :param vgg_t7_file Filename for the VGG pretrained net
    :param decode_t7_file Filename for the pretrained decoder net
    :param resize=[500,500] Size the images are resized to. Set to None for no resizing.
    '''
    with tf.Graph().as_default() as g, tf.Session(graph=g) as sess, tf.variable_scope(tf.get_variable_scope(), reuse=False) as scope:
        c, c_filename = image_from_file(g, 'content_image', size=resize)
        s, s_filename = image_from_file(g, 'style_image',size=resize)
        _, c_vgg = graph_from_t7(c, g, vgg_t7_file)
        _, s_vgg = graph_from_t7(s, g, vgg_t7_file)
        c_vgg = c_vgg[30]
        s_vgg = s_vgg[30]
        stylized_content = AdaIN(c_vgg, s_vgg, alpha)
        c_decoded, _ = graph_from_t7(stylized_content, g, decode_t7_file)
        c_decoded = postprocess_image(c_decoded)
        c = postprocess_image(c)
        s = postprocess_image(s)
        feed_dict = {c_filename: content, s_filename: style}
        combined, style_image, content_image = sess.run([c_decoded, s, c], feed_dict=feed_dict)
        return np.squeeze(combined), np.squeeze(content_image), np.squeeze(style_image)
        