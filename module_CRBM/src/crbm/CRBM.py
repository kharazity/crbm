import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import Initializer
import openfermion as of
from openfermion.ops.operators import SymbolicOperator, QubitOperator, FermionOperator
from openfermion.utils import hermitian_conjugated as hc
import numpy as np


class HFInitializer(Initializer):
    """ initializer class subtyping tf Initializer for CRBM"""



class CRBM(Model):
    """
    A class for initializing and training a complex-valued restricted boltzman machine. Following work from Torlai and an implementation by Andi Gu, we attempt to systematize this measurement procedure to execute in conjunction with quantum algorithms such as VQE, tomography, etc.
    Attributes:
        op_list : subtype of openfermion SymbolicOperator of operator we use for training RBM
        num_vis : int number of nodes in visible layer, equals num_qubits
        num_qubits : sugar for num_vis
        num_hid : int number of nodes in hidden layer
        training_params : dictionary of hyperparameters that affect the training
        optimizer_params : dictionary for optimization type and parameters
        state_vecs : tf.Tensor of quantum mechanical statevectors
    """

    def __init__(self, 
                H_op : QubitOperator,
                num_hid : int,
                weight_initializer : Initializer = tf.keras.initializers.RandomUniform,
                training_params : dict = {},
                optimizer_params : dict = {},
                save_best : bool = 1,
                save_path : str = ".",
                filename :str = "crbm"):
        """
        Args:
            H_op: a QubitOperator object (a transformed fermionOperator)

            num_hid: number of nodes in hidden layer of CRBM.

            weight_initialize: a subclass of tf Initializer that specifies how the weights of the CRBM should be initialized. Defaults to RandomUniform, various initializer types can be found here: https://www.tensorflow.org/api_docs/python/tf/keras/initializers; should not include object initializer '()'

            training_params: Dictionary for specifying hyperparameters for the training of the CRBM.
            Initializes to: {'min_loss': 1e-5, 'learning_rate' : 1e-1,'epochs': 1000, 'learning_patience': 100, 'lr_reduce': 0.3,
                                          'stop_patience': 100, 'n_samples': 1000, 'batch_size': 1000}

            optimizer_params: Dictionary for specifying the optimizer and parameters to pass to it. Valid keys are argument names to tf.optimizers.Optimizer()

            save_best: Save CRBM which performs best from those trained, otherwise all models are saved into a folder. Defaults to true.

            save_path: String specifying where to save the files to. Defaults to user's current directory.

            filename: String specifying name of saved model files
        Returns:
            CRBM object
        """
        super(CRBM, self).__init__()

        self.op = H_op
        self.num_vis = H_op.many_body_order() #num_qubits
        self.num_qubits = self.num_vis
        self.num_hid = num_hid

        #assert hermitian operator
        assert(hc(H_op) == H_op)
        
        assert(issubclass(weight_initializer, Initializer))
        self.weight_initializer = weight_initializer
        
        #initialize CRBM parameters
        init = self.weight_initializer
        self.vis_bias = tf.Variable(initial_value =
                                    self.complex_init(init, (self.num_vis, 1)), 
                                    dtype = tf.dtypes.complex64,
                                    trainable = True)
        self.hid_bias = tf.Variable(initial_value = 
                                    self.complex_init(init, (self.num_hid, 1)),
                                    dtype = tf.dtypes.complex64, 
                                    trainable = True)
        self.W = tf.Variable(initial_value = 
                            self.complex_init(init, (self.num_vis, self.num_hid)),
                            dtype = tf.dtypes.complex64, 
                            trainable = True)

        #initialize training parameter dict
        self.training_params = {'min_loss': 1e-5, 
                                'epochs': 1000,
                                'learning_rate' : 1e-1,
                                'learning_patience': 100, 
                                'lr_reduce': 0.3,
                                'stop_patience': 100,
                                'n_samples': 1000,
                                'batch_size': 1000,
                                'verbose' : False}
        #update training parameters based on user supplied argument
        for key in training_params:
            self.training_params[key] = training_params[key]

        #same for optimizer_params
        self.optimizer_params = {"optimizer": tf.keras.optimizers.Adam, "learning_rate": 1e-1}
        for key in optimizer_params:
            self.optimizer_params[key] = optimizer_params[key]
        
        state_vecs = np.array([list("{0:b}".format(x).zfill(self.num_vis)) 
                                for x in np.arange(2**self.num_vis)]).astype(np.int)
        self.state_vecs = tf.cast(tf.convert_to_tensor(state_vecs),
                                tf.dtypes.complex64)
        
        self.save_best = save_best
        self.save_path = save_path
        self.filename = filename
        
    #END __init__

    @staticmethod
    def complex_init(initializer :Initializer, shape :tuple) -> tf.complex:
        """
            Args: 
                initializer: a subclass of tf.keras.initializers.Initializer
                shape: tuple of ints for shape of tensor to initialize
            Returns:
                tensor of shape 'shape' initialized to complex values
        """
        init = initializer()
        re = init(shape = shape, dtype = tf.dtypes.float32)
        imag = init(shape = shape, dtype = tf.dtypes.float32)
        return tf.complex(re, imag)

    @property
    def parameters(self) -> tuple:
        return (self.vis_bias, self.hid_bias, self.W)

    def psi(self, state_v :tf.Tensor) -> tf.Tensor:
        """
        Args:
            complex unit vector or set of vectors as a tf.Tensor

        Returns: 
            crbm's generated wavefunction psi as a tf.Tensor
        """
        if state_v.ndim == 1:
            state_v = tf.expand_dims(state_v, -1)
        else:
            state_v = tf.transpose(state_v)
        
        vis_bias, hid_bias, W = self.vis_bias, self.hid_bias, self.W

        E = tf.linalg.adjoint(vis_bias) @ state_v + \
            tf.reduce_sum(tf.math.log(tf.math.conj(tf.exp(hid_bias + 
                        tf.linalg.adjoint(W) @ state_v)) + 1 ), axis = 0)

        return tf.transpose(tf.exp(E))

    def probability(self, measurements : np.ndarray) -> float:
        """
        Args: 
            np.ndarray of measurements over various operators
        Returns:
            the probability of the CRBM being able to produce psi from the measurements provided
        """
        
        psi_vec = self.psi(self.state_vecs)
        psi_vec /= tf.linalg.norm(psi_vec)
        psi_vec = tf.cast(psi_vec, tf.complex64)
        psi_proj = tf.math.conj(measurements) @ psi_vec
        return tf.math.abs(psi_proj)**2

    def call(self, measurements : np.ndarray):
        """
        overload of tf.keras.models.Model call method
        Args:
            measurements: np.ndarray of measurements over various operators
        
        Returns:
            singleton tf.Tensor of the loss from the measurement probabilities
        """
        self.add_loss(-tf.math.reduce_mean(
            tf.math.log(self.probability(measurements))))

    def measure(self, psi :np.array, meas_op :QubitOperator,
                n_sample :int = 100) -> np.array:
        """
        Args:
            psi: state vector like array
            meas_op: the qubit operator we wish to compute expectation values of
            n_sample: number of samples to take from probability distribution
        Returns:
            np.array of measurements
        """
        #assert(len(psi) == 2**self.num_qubits)
        mat = of.linalg.get_sparse_operator(meas_op, n_qubits = self.num_qubits).todense()
        _, eig_vecs = np.linalg.eigh(mat)
        eig_vecs = eig_vecs.astype(np.complex64)
        probs = np.conj(eig_vecs.T) @ psi
        probs = np.array(np.square(np.abs(probs))).flatten()
        if('n_samples' in self.training_params):
            n_sample = self.training_params['n_samples']
        #not 100% sure what this is doing
        N = self.num_qubits
        idx = np.random.choice(2**N, size = n_sample, p = probs)
        return eig_vecs.T[idx]
    
    #END CRBM class

class EarlyStoppingByLoss(Callback):
    """
        Stop training if loss drops below a specified value
    """
    def __init__(self, monitor :str = 'val_loss', 
                value :float = 1e-5, verbose :bool = 0):
        """
            Args: 
                monitor: string specifying which parameter to monitor
                value: float specifying minimum loss at which to terminate
                verbose: boolean for verbose output
        """
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
    def on_epoch_end(self, epoch :int, logs :dict = {}):
        current = logs.get(self.monitor)
        if current < self.value:
            print("Epoch %05d: early stopping THR" %epoch)
        self.model.stop_training = True
    #END EarlyStoppingByLoss

def train(crbm :Model, psi :np.array, seed :int = 0) -> np.array:
    """ 
    Args: 
        crbm: instantiated CRBM object
        psi: statevector like object as an np.array
        seed: random seed int
    Returns:
        normalized statevector produced by crbm
    """
    psi = psi.astype(np.complex64)
    np.random.seed(seed)
    op_terms = crbm.op.terms
    training_dict = crbm.training_params
    n_samples = training_dict['n_samples']
    measurement_list = [crbm.measure(psi, QubitOperator(op),
                        n_sample=n_samples) for op in op_terms]
    measurements = np.concatenate(measurement_list, axis = 0)
    batch_size = 20

    optimizer_dict = crbm.optimizer_params
    learning_rate = optimizer_dict['learning_rate']
    optimizer = optimizer_dict['optimizer']


    min_val = training_dict['min_loss']
    verbose = training_dict['verbose']
    loss_patience = training_dict['stop_patience']
    lr_patience = training_dict['learning_patience']
    lr_factor = training_dict['lr_reduce']
    batch_size = training_dict['batch_size']
    save_best = crbm.save_best
    crbm.compile(optimizer = optimizer(learning_rate))
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint('model.tf', monitor = 'loss', verbose = verbose, save_best_only = save_best),
        EarlyStoppingByLoss(monitor = 'loss', value=min_val),
        tf.keras.callbacks.ReduceLROnPlateau(patience = lr_patience, monitor = 'loss', factor = lr_factor),
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience = loss_patience, restore_best_weights=True)]
    hist = crbm.fit(measurements, verbose = verbose, callbacks=callbacks, batch_size = batch_size, shuffle=False)
    #update for custom string in case user passes a save_path or filename
    filename = crbm.filename
    crbm.save_weights(filename)
    crbm.load_weights(filename)
    psi2 = crbm.psi(crbm.state_vecs).numpy().flatten()
    psi2 *= 1/psi2[0]
    print(n_samples, seed, min(hist.history['loss']), len(hist.history['loss']))
    return psi2/np.linalg.norm(psi2)
    #END train