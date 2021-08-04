import grpc
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import tensorflow as tf

def create_grpc_stub(host, port=8500):
    hostport = "{}:{}".format(host, port)
    channel = grpc.insecure_channel(hostport)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    return stub

def grpc_request(stub, data_sample, model_name='my_model', signature_name='classification'):
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_name
    request.model_spec.signature_name = signature_name
    request.inputs['inputs'].CopyFrom(tf.make_tensor_proto(data_sample, shape=[1,1]))
    result_future = stub.Predict.future(request, 10)
    return result_future

host = "http://localhost"
payload = {'age': 39,
           'workclass': 'State-gov', 
           'fnlwgt': 77516, 
           'education': 'Bachelors', 
           'education-num':13,
           'marital-status': 'Never-married', 
           'occupation': 'Adm-clerical', 
           'relationship': 'Not-in-family', 
           'race': 'White', 
           'sex': 'Male',
           'capital-gain': 2174, 
           'capital-loss': 0, 
           'hours-per-week': 40, 
           'native-country': 'United-States'
           }
stub = create_grpc_stub(host, port=8500)
rs_grpc = grpc_request(stub, payload)