import predict
import argparse
import get_data
import train_gen
import train_dis
import write_tfrecords
import data_distributions


# Main function to run the code files as needed
def main(args):
    if not args.strtopt == 'p' and (args.strtopt == 'a' or input("Get UTKFace Dataset? (y/n): ") == 'y'):
        data = get_data.get_data(args.job_dir)
        write_tfrecords.write_tfrecords(data, args.job_dir)
    if not args.strtopt == 'p' and (args.strtopt == 'a' or input("Display data distributions (y/n): ") == 'y'):
        data_distributions.getDataDetails(data)
    if not args.strtopt == 'p' and (args.strtopt == 'a' or input("Run train_dis.py? (y/n): ") == 'y'):
        train_dis.train_discriminator(
            args.job_dir,
            args.epochs,
            args.learning_rate,
            args.patience)
        train_gen.train_generator(
            args.job_dir,
            args.epochs,
            args.learning_rate,
            args.patience)
    if not args.strtopt == 'p' and (args.strtopt == 'a' or input("Run train_gen.py? (y/n): ") == 'y'):
        train_gen.train_generator()
    if args.strtopt == 'a' or args.strtopt == 'p' or input("Run predict.py? (y/n): ") == 'y':
        predict.predict(args.job_dir, args.prediction_file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Options")
    parser.add_argument(
        "-so",
        "--strtopt",
        type=str,
        default="",
        choices=["a", "p", ""],
        help="a -> Run All: Make, Train, and Predict")
    parser.add_argument(
        "-jb",
        "--job_dir",
        type=str,
        default="..",
        help="Base Directory")
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs")
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=0.0002,
        help="Learning Rate for Adam")
    parser.add_argument(
        "-p",
        "--patience",
        type=int,
        default=20,
        help="Early Stopping Patience")
    parser.add_argument(
        "-pf",
        "--prediction_file_name",
        type=str,
        default="/UTKFace/8_1_0_20170109203324490.jpg.chip.jpg",
        help="Image for prediction")
    args = parser.parse_args()
    main(args)
