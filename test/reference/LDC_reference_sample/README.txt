Corpus Title:    CCU TA1 Mandarin/Chinese Development Annotation Sample
LDC Catalog-ID:  LDC2022E12
Date:            July 13, 2022


1. Introduction

This package contains a sample of annotation output for the CCU TA1
data. The purpose of this package is to illustrate the annotation output
format. The actual annotations in this package are not actual annotation
data, but they do illustrate the approach that is being taken for
annotating TA1 data. 

(Note: The TA2 dev data annotation will follow a similar format, but some
aspects of TA2 annotation are still being finalized so there may be minor
differences in approach).

TA1 annotation consists of four tasks, conducted over the same set of TA1
data:
-valence and arousal annotation
-emotion category annotation 
-norm category annotation 
-changepoint annotation

The results of each annotation task are presented in a tab-delimited output
file, described below. 

Valence/arousal, emotion and norm annotation are performed on one segment
at a time. Segments have been defined simply as a convenience for annotation
and should not be assumed to reflect any meaningful unit of discourse,
although their duration may roughly resemble a turn or utterance. In
addition to the annotation tab files, the annotation release packages will
contain a segments.tab file which defines the segment boundaries used for
annotation. Changepoint annotation is performed without respect to segment
boundaries.

We obtain up to 3 independent annotator judgments per segment in the
valence/arousal and emotion category annotation tasks, while norm
annotation and changepoint annotation have a single annotator for each
segment.

TA1 annotation will be released incrementally. Note that each incremental
release will reflect a complete snapshot of annotation completed to
date. This means that some files and segments will be "incomplete" in a
given data release; that is, a segment may have judgments for some tasks
and not others, and a segment that is subject to multi-way annotation may
have only one-way annotation present in a given release package. It is also
important to note that annotation labels on a given segment may change
between releases as additional quality review is conducted. Each annotation
release package should be treated as a complete replacement for the one
that precedes it.

Additional details about the package structure and contents appear below.

2. Directory Structure and Content Summary

The directory structure and contents of the package are summarized
below -- paths shown are relative to the base (root) directory of the
package:

  data/                 -- contains a tab-delimited table file for each
                           annotation type: emotion; norms;
                           valence and arousal; changepoint
                           see content description below for details

  docs/                 -- contains a tab-delimited table file with
                           information about start/end of each annotation
                           segment

2.1 Content Summary

All annotations are delivered as tab-delimited files; each type of
annotation file is described below. A segments.tab file is included in
the /docs directory to provide a sample format of the segmentation
information that will be provided in upcoming releases.

No other documents are provided with this annotation format sample.
Guidelines and any other relevant documentation about the annotation
process and rules will be provided in upcoming annotation releases.

3. Annotations

Emotion and valence/arousal are 3-way annotations. Norm and changepoint
annotation are judged by a single annotator, i.e. are not multi-way
annotations. For multi-way annotation, each row in the table represents a
unique annotator and segment combination, i.e. a segment will have 3 rows
to show 3 independent annotations when annotation for that segment is
complete. All current annotations will be included in the release, including
segments for which 3-way annotation or complete file annotation is not yet
complete.

3.1 "emotions.tab" -- emotion category annotation

This table contains the results of emotion category annotation. Segments
that have been annotated are included in the table. The columns are
tab-delimited and the initial line of the file provides the column labels
as shown below:

  Col.# Content
  1. user_id            - annotator user ID
  2. file_id            - 9-character source document unique identifier
  3. segment_id         - unique identifier for the segment consisting of
                          the file_id appended by a 4-digit within-file segment
                          ID number (e.g. M010009BD_0001)
  4. emotion            - emotion category present in segment (fear, anger, sadness,
                          joy, disgust, surprise, trust, anticipation, none or
                          nospeech)
                          When an annotator judges a segment to contain
                          multiple emotion categories, this field will contain a 
                          comma-separated list of values.
                          Annotators select 'none' to indicate that no emotions
                          are present in the segment. 'nospeech' indicates
                          that the segment was manually or automatically
                          judged to contain no speech. 
  5. multi_speaker      - annotation to indicate if multiple speakers are
                          expressing emotion in a segment: TRUE, FALSE, nospeech,
                          or EMPTY_NA for segments annotated 'none'

3.2 "norms.tab" -- norm category annotation

This table contains norm category annotation. Segments that have been
annotated are included in the table. The columns are tab-delimited and the
initial line of the file provides the column labels as shown below:

  Col.# Content
  1. user_id            - annotator user ID
  2. file_id            - 9-character source document unique identifier
  3. segment_id         - unique identifier for the segment consisting of
                          the file_id appended by a 4-digit within-file segment
                          ID number (e.g. M010009BD_0001)
  4. norm               - norm category present in segment, norms are assigned a 3-digit
                          norm_id (001, 002, 003, etc., none, or nospeech)
                          NOTE: Annotators select 'none' to indicate that no observable
                          norms are present in the segment.
  5. status             - status of annotated norm, indicating if the norm is
                          adhered to or violated (adhere, violate, nospeech, or EMPTY_NA).

Note that if a given segment has multiple observed norms, the segment
will appear in the table multiple times (one row for each observed norm).

3.3 "valence_arousal.tab" -- valence and arousal annotation

This table contains valence and arousal annotation. Segments that have been
annotated are listed in the table. The columns are tab-delimited and the initial
line of the file provides the column labels as shown below:

  Col.# Content
  1. user_id            - annotator user ID
  2. file_id            - 9-character source document unique identifier
  3. segment_id         - unique identifier for the segment consisting of
                          the file_id appended by a 4-digit within-file segment
                          ID number (e.g. M010009BD_0001)
  4. valence_continuous - valence value annotated by continuous slider (1-1000 or nospeech)
  5. valence_binned     - scalar "bin" of valence value
                          (1, 2, 3, 4, 5 or nospeech)
                          values 1-200 = 1
                          values 201-400 = 2
                          values 401-600 = 3
                          values 601-800 = 4
                          values 801-1000 = 5
                          1 is most negative valence and 5 is most positive valence
                          binned valence is automatically derived from the value in
                          column #4
  6. arousal_continuous - arousal value annotated by continuous slider (1-1000 or nospeech)
  7. arousal_binned     - scalar "bin" of arousal value
                          (1, 2, 3, 4, 5 or nospeech)
                          values 1-200 = 1
                          values 201-400 = 2
                          values 401-600 = 3
                          values 601-800 = 4
                          values 801-1000 = 5
                          1 is the lowest arousal and 5 is highest arousal
                          binned arousal is automatically derived from the value in
                          column #6

3.4 "changepoint.tab" -- change point annotation

This table contains changepoint annotation. The columns are tab-delimited and the
initial line of the file provides the column labels as shown below:

  Col.# Content
  1. user_id            - annotator user ID
  2. file_id            - 9-character source document unique identifier
  3. timestamp          - timestamp (in seconds), for audio or video data,
                          or character offset, for text data, of the changepoint
                          (e.g. 20.982 or 75)
  4. impact_scalar      - scalar judgment of the changepoint's impact on the
                          conversation (1, 2, 3, 4 or 5)
                          1 is the most negative impact and 5 is the most positive
                          impact
  5. comment            - brief annotator description of why they indicated the
                          changepoint*

*The comment is raw free text, as provided by annotators and has not been modified
or corrected for release.

4. Documentation

"segments.tab" -- segmentation information the portion of the document
subject to annotation

The columns are tab-delimited and the initial line of the file provides the
column labels as shown below:

  1. file_id            - 9-character source document unique identifier
  2. segment_id         - unique identifier for the segment consisting of
                          the file_id appended by a 4-digit within-file segment
                          ID number (e.g. M010009BD_0001)
  3. start              - start timestamp (in seconds), for audio or video data,
                          or character offset, for text data, of the segment
                          (e.g. 20.982 or 75)
  4. end                - end timestamp (in seconds), for audio or video data,
                          or character offset, for text data, of the segment
                          (e.g. 345.67 or 1035)

5. Copyright Information

   (c) 2022 Trustees of the University of Pennsylvania

6. Contact Information

Stephanie Strassel <strassel@ldc.upenn.edu> CCU PI
Dana Delgado <foredana@ldc.upenn.edu> CCU Project Manager
Summer Ploegman <sawp@ldc.upenn.edu> CCU Annotation Manager

----
README created by Dana Delgado July 11, 2022
README updated by Dana Delgado July 13, 2022
