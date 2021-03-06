graph = tf.Graph()

features = np.array([[6.83324017e-02, 4.55211316e-01,-1.41892820e-01, 6.41751984e-01, -5.45895865e-01, 5.38657679e-01, 1.93379897e-01, 1.60154529e-01, 1.57859872e-02, 1.36758294e-02, 4.40859703e+00, 4.96067050e+03, -5.95230431e+01, 2.29624126e+00, 4.02069655e+00], [8.82284599e-01, 6.42900440e-01,-4.27639642e-02, 1.83567706e-01, 7.52404702e-01, -6.32605771e-01, 5.40391531e-01, 5.84584613e-01,-7.15044264e-03,-8.23328268e-02, 6.29273115e+00,-4.32369561e+01, 7.07259958e+00,-1.02810233e+00,-7.04034886e-01], [5.48660773e-01, 8.08794529e-01,-5.96924524e-02,-7.26052964e-01,-2.70772000e-02, 6.87105464e-01, 5.68913359e-01, 4.76252594e-01, 4.14203699e-02,-5.79935485e-03, 9.40232256e+00,-2.01665599e+04, 1.34500232e+01,-2.24989629e-01, 2.52753983e-01], [7.46613308e-01, 8.23272733e-01,-1.04753678e-01, 7.87653516e-01, 5.33736860e-01, 3.07777360e-01, 8.51814816e-01, 7.29870149e-01,-5.67521706e-03, 2.37203887e-02, 6.33280960e+00, 4.08845288e+05, 4.48007235e+01, 5.33139458e-02, 2.37384134e-02], [4.47498908e-01, 1.49080014e-01,-9.07106172e-03,-2.67174181e-01,-5.21700457e-01, 8.10213916e-01, 9.18038857e-01, 8.36740457e-01,-7.64173908e-03,-1.18870530e-02, 6.18394833e+00, 7.37307204e+01,-5.58432681e+01, 3.83996968e-01, 9.18497562e-01], [4.71607629e-01, 1.31179570e-01,-4.56846546e-02,-9.27597302e-01,-3.63639607e-01,-8.56123912e-02, 3.32925650e-01, 2.86999292e-01,-1.37396795e-01,-2.39745171e-01, 6.28318531e+00,-9.03421275e+04,-9.83543039e+03,-1.09839821e+00, 1.05041514e+00], [4.71613040e-01, 1.31166299e-01,-4.56797268e-02,-9.27775404e-01,-3.64117510e-01,-8.15551274e-02, 3.32854008e-01, 2.86979856e-01,-1.36950051e-01,-2.39623484e-01, 6.28318531e+00,-2.85787226e+05, 1.02588457e+05,-1.09795489e+00, 1.05020120e+00], [1.72510574e-01, 3.40244123e-02,-1.78258372e-01,-1.78623912e-01, 9.82406854e-01,-5.45001987e-02, 6.49133952e-01, 4.58514334e-01,-1.05587941e-01,-1.50382361e-01, 6.56445597e+00,-7.39915259e+01,-3.39043636e+01, 8.32312454e-01, 1.66266815e+00]])

labels = np.array([[ 1., 0.], [ 0., 1.], [ 0., 1.], [ 1., 0.], [ 0., 1.], [ 0., 1.], [ 0., 1.], [ 1., 0.]])

totalLoss = 0
totalTest = 0

with graph.as_default():
    x = tf.placeholder("float", [None, 15], name = "x")
    y = tf.placeholder("int64", [None, 2], name = "y")

    h1 = tf.Variable(tf.truncated_normal([15, 100], stddev = 0.1), name = "h1") 
    out = tf.Variable(tf.truncated_normal([100, 2], stddev = 0.1), name = "out")
    b1 =  tf.Variable(tf.truncated_normal([100], stddev = 0.1), name = "b1")
    bout = tf.Variable(tf.truncated_normal([2], stddev = 0.1), name = "bout")


    def model(x):
        layer_1 = tf.add(tf.matmul(x, h1), b1)
        layer_1 = tf.nn.relu(layer_1)

        out_layer = tf.matmul(layer_1, out) + bout
        return out_layer

    logits = model(x)
    
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = y))
    optimizer = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(loss)

    with tf.Session(graph = graph) as session:
        for i in range(10):
            tf.global_variables_initializer().run()
            
            _ = session.run(optimizer, feed_dict = {x : features, y : labels})
            l = session.run(loss, feed_dict = {x : features, y : labels})
            test = session.run(tf.reduce_mean(-tf.reduce_sum(labels * tf.log(tf.nn.softmax(logits) + 1e-10), 1)), feed_dict = {x : features})
            totalLoss += l 
            totalTest += test

print("mathematical : ", totalTest *1. /10)
print("sparse_softmax_cross_entropy : ", totalLoss *1. /10)
