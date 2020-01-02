# Recommender

[![Maintainability](https://api.codeclimate.com/v1/badges/ffd49b28c31a2e188ce0/maintainability)](https://codeclimate.com/github/matt-bernhardt/recommender/maintainability)

This started as a very rudimentary recommender system that was built during a
two-day Learning Days sprint at the MIT Libraries. It is currently being
extended and refactored as time allows.

The process of using the tool is currently fairly simple, and should be done
in a virtualenv (see `requirements.txt` for packages).

## Requirements

These tools assume a set of JSON files exist in the `data/` directory. An
example can be found at `data/samples/sample.json`. This is the structure of
a JSON record from our ArchivesSpace instance, as processed by the [Mario
pipeline](https://github.com/mitlibraries/mario).

## Data preparation steps

The first three scripts will probably be converted into one preparation step,
and there is a part of this process that was done originally in [OpenRefine](https://openrefine.org/), so
is not represented here.

* `1gather.py` will combine that list of JSON files into one file.
* `2prune.py` will reconstitute the list without subject fields (which will
  hopefully be used to verify that the recommender is functioning correctly)
* `3rectangular.py` attempts to flatten the JSON structure into something
  more rectangular.

**Please note:** during the original sprint, the output of these scripts was
then processed further in OpenRefine, with sacrifices made to the point where
only the notes field of the original JSON was preserved.

At the end of the preparation phase, you should have one document with a
single text field per record. The contents of that text field should be the
concatenated notes field from a single record - so if you start with 1,000
ASpace records, you end with 1,000 text fields.

## Training and operation steps

The final two steps concern themselves with training, and then applying, the
recommender system.

* `4lda-train-and-save.py` creates the various files needed for the system to
  function. These are saved in binary formats in the `data/output/` directory.
* `4lda-load-and-recommend.py` loads these binary files and then conducts a
  search using the contents of the `trial` variable that is defined within the
  script. The results of this search are displaed on in the terminal window.

## Next steps

Yikes, there is a lot of work that could be done to clean this up. Pick a part
of this repository and there are probably a multitude of things that could be
done to improve the workflow, make the math more robust, preserve the data
in a more usable way, or generally make things better.
