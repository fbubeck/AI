?($	?0&????2dv;?o??I.?!????!|??Pk???$	???0??@??Y??@?8 ?-t@!Ա?0&@4@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$I.?!?????ݓ??Z??Az6?>W[??Y?n??ʱ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?	h"lx??&S????A~??k	???Y46<?R??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&aTR'?????0?*??A2??%????YHP?s??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?q??????"?uq??A ?o_???Y?|a2U??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?6?[ ?????9#J??A o?ŏ??Yw-!?l??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?ܵ?|????U??????A-??????Y/n????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&$(~?????^)???AгY?????Y?St$????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??g??s????St$???A}??b???Y????<,??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&D????9?????{????Ax$(~??Y??ׁsF??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	-??????A?c?]K??A+??????Y	?^)ː?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
h"lxz???x??#????A??x?&1??Ye?X???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&L7?A`????.n????AOjM???Ye?X???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&4??7?????? ?rh??A???????Y^K?=???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?d?`TR?????????Aq=
ףp??Y?e??a???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&&S??????HP??AY?? ???YV????_??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?_vO??q???h ??Aj?t???Y]m???{??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????S???e?c]???A|a2U0*??Y?k	??g??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&|??Pk????K7?A`??A??ܵ?|??Y/?$???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??7??d??}гY????Ah"lxz???Y?e??a???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?JY?8???Ǻ????A?????Y6?;Nё??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&X?2ı.??????(??Ad?]K???Y?? ?rh??*	???????@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?d?`TR??!??~UA@) ?o_???1k?c/?@@:Preprocessing2F
Iterator::Model? ?	???!n?vRǱA@)i o????1<tu??5@:Preprocessing2U
Iterator::Model::ParallelMapV2u????!>?_?+@)u????1>?_?+@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?JY?8???!ɀ?V'P@)$????۷?1E?vI?%@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?46<??!J????y*@):??H???13??cM?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?w??#???!cS?w?@)?w??#???1cS?w?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?????!??ب"?2@)?-????1!???TM@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*0*??D??!???A>?@)0*??D??1???A>?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 50.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?gn??@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	^bi!???& ?$,???ݓ??Z??!?K7?A`??	!       "	!       *	!       2$	:??NZ???Vs?????d?]K???!h"lxz???:	!       B	!       J$	ZOl?ւ????)???	?^)ː?!?n??ʱ?R	!       Z$	ZOl?ւ????)???	?^)ː?!?n??ʱ?JCPU_ONLYY?gn??@b Y      Y@q?N?+)D@"?
both?Your program is POTENTIALLY input-bound because 50.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?40.3217% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 