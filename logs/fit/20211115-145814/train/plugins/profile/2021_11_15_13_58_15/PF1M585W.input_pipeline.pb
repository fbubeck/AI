$	?l؈??????n6????C?l??!??3????$	2aP ?s@W?+???@u?0??K@!??#??'1@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$yX?5?;???7??d???A??h o???Y2??%䃮?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??3?????/L?
F??A?lV}???Y?R?!?u??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?H?}8??$(~??k??Al	??g???Y?:pΈ??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???????	?^)???A+??ݓ???Y?0?*??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?(\??????#??????A?L?J???Y	?^)ː?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&@?߾?????j+????A???{????Y?<,Ԛ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?l??????O??e???A[Ӽ???Y?j+??ݓ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?8EGr?????/?$??A???镲??Ye?X???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&T㥛? ????ZӼ???A??D????Yy?&1???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	?/?'????Pk?w??A?H?}8??Y?W[?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
?f??j+???S㥛???A??m4????Y???{????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????`??"????A	??g????YjM????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?? ?	???j+?????A???Q???Y???????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?{??Pk??X9??v??A??ܵ???YV-???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????H??l	??g???Ah"lxz???YjM????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?G?z????0?*??Aj?q?????Y??y?):??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????߾???g??s???A?Y??ڊ??Yvq?-??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&M?O??????_vO??Aq=
ףp??Yc?ZB>???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?St$????l	??g???A???????Y???QI??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&lxz?,C??_?L?J??A?HP???Y+??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??C?l????ZӼ???A\ A?c̽?Y?{??Pk??*	?????Ɇ@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??(????!??????A@)؁sF????1ߓ?0?@@:Preprocessing2F
Iterator::ModelA??ǘ???!??a?v@@)㥛? ???1B?9??4@:Preprocessing2U
Iterator::Model::ParallelMapV2??????!?E??!?)@)??????1?E??!?)@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipvOjM??!z?3O??P@)??|гY??1?ƈ???&@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSliceW[??재?!??????@)W[??재?1??????@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate?_?L??!	???w.@)??q????1??;??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?B?i?q??!W????3@)"??u????12?????@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*????<,??!~?ۜ@)????<,??1~?ۜ@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 55.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9m?Ǯ?@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	?EC[?]???j???O????ZӼ???!?/L?
F??	!       "	!       *	!       2$	"V??E??Ӎ?0?e??\ A?c̽?!??ܵ???:	!       B	!       J$	?wa????????p??y?&1???!2??%䃮?R	!       Z$	?wa????????p??y?&1???!2??%䃮?JCPU_ONLYYm?Ǯ?@b 