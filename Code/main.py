import predict
import argparse
import get_data
import train_gan
import write_tfrecords
import data_distributions


def main(args):
	# Defines the main function to run the code files as needed
	data = None
	if not args.strtopt == 'p' and (args.strtopt == 'a' or input("Get UTKFace Dataset? (y/n): ") == 'y'):
		data = get_data.get_data(args.job_dir)
		write_tfrecords.write_tfrecords(data, args.job_dir)
	if not args.strtopt == 'p' and (args.strtopt == 'a' or input("Display data distributions (y/n): ") == 'y'):
		data_distributions.get_data_details(data)
	if not args.strtopt == 'p' and (args.strtopt == 'a' or input("Train the GAN model? (y/n): ") == 'y'):
		train_gan.train(args.batch_size_per_replica, args.gen_lr, args.disc_lr, args.epochs, args.patience,
		                args.job_dir)
	if args.strtopt == 'a' or args.strtopt == 'p' or input("Run predict.py? (y/n): ") == 'y':
		predict.predict(args.prediction_file_name, args.job_dir)


if __name__ == "__main__":
	# Gets the arguments passed by the user and runs the remaining code files accordingly
	parser = argparse.ArgumentParser(description="Options")
	parser.add_argument(
		"-so",
		"--strtopt",
		type=str,
		default="",
		choices=["a", "p", ""],
		help="a -> Run All: Make, Train, and Predict, p -> Predict")
	parser.add_argument(
		"-jb",
		"--job_dir",
		type=str,
		default="..",
		help="Base Directory")
	parser.add_argument(
		"-bspr",
		"--batch_size_per_replica",
		type=int,
		default=32,
		help="Batch Size per GPU")
	parser.add_argument(
		"-e",
		"--epochs",
		type=int,
		default=100,
		help="Number of epochs")
	parser.add_argument(
		"-glr",
		"--gen_lr",
		type=float,
		default=0.00001,
		help="Learning Rate for the Generator")
	parser.add_argument(
		"-dlr",
		"--disc_lr",
		type=float,
		default=0.00001,
		help="Learning Rate for the Discriminator")
	parser.add_argument(
		"-p",
		"--patience",
		type=int,
		default=1000,
		help="Early Stopping Patience")
	parser.add_argument(
		"-pf",
		"--prediction_file_name",
		type=str,
		default="/UTKFace/8_1_0_20170109203324490.jpg.chip.jpg",
		help="Image for prediction")
	args = parser.parse_args()
	main(args)
