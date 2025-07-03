import csv

def data_processing(dataset_text_path, output_file):
    with open(dataset_text_path, encoding='utf-8') as f:
        f_lines = f.readlines()

    data = [['post_text', 'image_id', 'label']]
    for line in f_lines:
        if "\n" in line:
            line = line[0:-1]
        line_list = line.split('\t')

        if len(line_list) != 7:
            print(line_list)
            continue
        if line_list[6] == "fake":
            label = 0
        elif line_list[6] == "real":
            label = 1
        else:
            continue

        text = ' '.join([part for part in line_list[1].split() if not 'http' in part.lower()])

        data_cell = [text, line_list[3], label]
        data.append(data_cell)

    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(data)

if __name__ == '__main__':
    devset_text_path = './tw_dataset/twitter_dataset/devset/posts.txt'
    devset_output_file = './tw_dataset/twitter_dataset/devset/posts.csv'
    data_processing(devset_text_path, devset_output_file)

    testset_text_path = './tw_dataset/twitter_dataset/testset/posts_groundtruth.txt'
    testset_output_file = './tw_dataset/twitter_dataset/testset/posts_groundtruth.csv'
    data_processing(testset_text_path, testset_output_file)
