import tensorflow as tf 
import os 

class restore_model(object):
	"""docstring for run_model"""
	def __init__(self, model_folder, model_name):
		super(restore_model, self).__init__()
		self.model_name=os.path.join(model_folder, model_name)
		self.model_folder=model_folder

	def run_model(self, feature):
		with tf.Session() as sess_run:
			new_saver=tf.train.import_meta_graph(self.model_name)
			new_saver.restore(sess_run, tf.train.latest_checkpoint(self.model_folder+'/.'))

			graph=tf.get_default_graph()
			feature_in_restore=graph.get_tensor_by_name('feature_in:0')
			predict_restore=graph.get_tensor_by_name('predict:0')
			feed_dict={feature_in_restore: feature}
			prediction=sess_run.run([predict_restore], feed_dict)
		return prediction
