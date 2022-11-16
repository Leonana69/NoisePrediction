import numpy as np

def main():
    input_size = 32
    ref_noise = np.loadtxt("./data/ref_noise.txt", delimiter=',')
    tar_noise = np.loadtxt("./data/tar_noise.txt", delimiter=',')

    length = ref_noise.shape[0]
    print(ref_noise.shape)

    input = []

    for index in range(0, length - input_size):
        input.append(ref_noise[index:index + input_size])

    input = np.array(input)
    target = np.expand_dims(tar_noise[input_size:], 1)
    print(input.shape)
    print(target.shape)

    with open('./data/train.npy', 'wb') as f:
        np.save(f, input)
        np.save(f, target)

    return 0


if __name__ == "__main__":
    main()