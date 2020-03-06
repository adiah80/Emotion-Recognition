
from lib import *
from model import CNNModel
from dataset import EmotionDataset

emotion_labels = np.array(["DIS", "FEA", "HAP", "NEU", "SAD"])


############################### MAIN ###############################

if __name__ == "__main__":

	# Parse arguments.

	parser = argparse.ArgumentParser()

	parser.add_argument(
		"--test-location", 
		"-t", 
		required=True,
		type=str,
		help="Location of the test folder containg files for prediction."
	)

	args = parser.parse_args()
	test_root = args.test_location

	# Get the test file_paths from the specified path.
	test_paths = []

	for file_name in os.listdir(test_root):
		file_path = os.path.join(test_root, file_name)
		test_paths.append(file_path)
	
	test_paths = np.array(test_paths)
	test_paths.sort()
	test_paths_c = test_paths
	
	# Generate the test data loader.
	batch_size = 128
	test_dataset = EmotionDataset(paths_c=test_paths_c, mode="test", transform = transforms.ToTensor())
	test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

	model = CNNModel()

	# Specify model to load.
	# Weights specified in 'Weights' folder
	model_no = 22
	iterr = 5800
	# model.load_state_dict(torch.load('Weights/model{}_iterr={}_state_dict.pt'.format(model_no, iterr), 
	#                                   map_location=torch.device('cpu')))

	model.load_state_dict(torch.load("weights.pt",  map_location=torch.device('cpu')))

	# Initialize final output arrays.
	file_paths_list = np.array([])
	preds_list = np.array([])

	for i, (images, file_paths) in enumerate(test_loader):
		
		images = Variable(images)

		# Evaluate the test data.
		model.eval()		
		outputs = model(images).detach()

		# Map predictions to meotions.
		preds = emotion_labels[np.argmax(outputs, axis=1)]

		file_paths_list = np.concatenate((file_paths_list, file_paths))
		preds_list = np.concatenate((preds_list, preds))


	output_df = df = pd.DataFrame({
						"File name" : file_paths_list,
						"prediction" : preds_list})

	# Save the output DataFrame as a text file.
	output_df.to_csv("output.txt", index=False)

	# Show the Dataframe.
	print(output_df)