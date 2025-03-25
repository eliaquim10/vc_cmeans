import os
import json
from datasets import GeneratorBasedBuilder, SplitGenerator, Value, Features, Sequence, DatasetInfo
from tqdm import tqdm
import mloader

class PopMangaTest(GeneratorBasedBuilder):
    def _info(self):
        return DatasetInfo(
            features=Features({
                "image_path": Value("string"),
                "magi_annotations": {
                    "bboxes_as_x1y1x2y2": Sequence(Sequence(Value("float32"))),
                    "labels": Sequence(Value("int32"))
                },
                "character_clusters": Sequence(Value("int32")),
                "text_char_matches": Sequence(Sequence(Value("int32"))),
                "text_tail_matches": Sequence(Sequence(Value("int32"))),
                "text_classification": Sequence(Value("int32")),
                "character_names": Sequence(Value("string"))
            })
        )

    def _split_generators(self, dl_manager):
        extracted_annotations_dir = dl_manager.download_and_extract("annotations.zip")
        extracted_annotations_dir = extracted_annotations_dir.replace("\\", "/") 
        
        
        mangaplus_links_path = os.path.join(extracted_annotations_dir, "annotations/mangaplus_links.txt")
        with open(mangaplus_links_path, "r") as fh:
            mangaplus_links = [x.strip() for x in fh.readlines()]
        for link in tqdm(mangaplus_links):
            os.system(f"mloader {link} --raw --chapter-subdir -o {extracted_annotations_dir}/images")
        return [
            SplitGenerator(
                name="seen",
                gen_kwargs={
                    "path_to_images": os.path.join(extracted_annotations_dir, "images"),
                    "annotations_dir": os.path.join(extracted_annotations_dir, "annotations"),
                    "split": "seen",
                }
            ),
            SplitGenerator(
                name="unseen",
                gen_kwargs={
                    "path_to_images": os.path.join(extracted_annotations_dir, "images"),
                    "annotations_dir": os.path.join(extracted_annotations_dir, "annotations"),
                    "split": "unseen",
                }
            )
        ]

    def _generate_examples(self, path_to_images, annotations_dir, split):
        split_file = os.path.join(annotations_dir, f"{split}.txt")
        path_to_images = annotations_dir.replace("\\", "/")
        annotations_dir = annotations_dir.replace("\\", "/")

        with open(split_file, "r") as fh:
            image_paths = [x.strip() for x in fh.readlines()]

        for idx, image_path in enumerate(image_paths):
            try:
                annotations_path = os.path.join(annotations_dir, image_path + ".json")
                image_path = os.path.join(path_to_images, image_path)
                # print(image_path)

                with open(annotations_path, "r") as fh:
                    annotations = json.load(fh)
                magi_annotations = annotations["bbox_annotations"]
                character_clusters = [int(x) for x in annotations["character_clusters"]]
                text_char_matches = annotations["text_char_matches"]
                text_tail_matches = annotations["text_tail_matches"]
                text_classification = annotations["text_classification"]
                character_names = annotations["character_names"]

                yield idx, {
                    "image_path": image_path,
                    "magi_annotations": magi_annotations,
                    "character_clusters": character_clusters,
                    "text_char_matches": text_char_matches,
                    "text_tail_matches": text_tail_matches,
                    "text_classification": text_classification,
                    "character_names": character_names
                }
            except FileNotFoundError as e:
                # raise Exception('Invalid json: {}\n {}'.format(e, annotations_path)) from None
                print("->", annotations_path)
                # exit()
        print("*"*100)