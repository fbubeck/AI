?($	d??m????&ao?????v????!???~?:??$	=?(??@}?>??@j??J?	@!Ө7???0@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??q????v??????A|??Pk???Y?	h"lx??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&q???h????D????A??HP??Y??q????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&gDio????yX?5?;??AL7?A`???Y	?c???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??n????+????A??B?i???Y?sF????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&<?R?!?????JY?8??AA??ǘ???YtF??_??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????S??V-?????A!?rh????Yy?&1???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??ͪ?????镲q??A????o??Y??0?*??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&C??6??Έ?????A???Q???Y??Ɯ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???镲??<Nё\???A??????Y?D???J??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	??ܵ??j?t???A?G?z??Y?4?8EG??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
??o_??z6?>W??AKY?8????Y???QI??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Q?|a2??<Nё\???A??	h"l??Y??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???镲??P??n???A䃞ͪ???Y??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	?c?Z??"?uq??A{?/L?
??Y-C??6??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?=yX?5??Y?? ???A+??????YJ+???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&*:??H??n4??@???A'1?Z??Y%u???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&bX9????)\???(??A?QI??&??Y??@??ǘ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?_vO??:#J{?/??A@a??+??Yw-!?l??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??:M???+??	h??Au????Y?0?*??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???~?:??0*??D??A+??ݓ???Y??ͪ?զ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??v????/?$???A?X?? ??Y䃞ͪϥ?*	????̠?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatsh??|???!?rA@)k+??ݓ??1?ڛ2?M?@:Preprocessing2F
Iterator::Model?s?????!uYN?'?B@)??/?$??1?}ug??5@:Preprocessing2U
Iterator::Model::ParallelMapV2.???1???!jNq?.@).???1???1jNq?.@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??????!????]O@)9??m4???1_??K'@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlicep_?Q??!?g??@)p_?Q??1?g??@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate1?*?Թ?!?Z?r?)@)??MbX??1yM?ͫ@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(~??k	??!?v"??0@)??ܵ?|??1 U??W@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*?+e?X??!?uB??$@)?+e?X??1?uB??$@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 55.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9n??f3@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	?i??g???$:??S??v??????!0*??D??	!       "	!       *	!       2$	??Й?&??8???k3???X?? ??!+??ݓ???:	!       B	!       J$	?f#U?`???י??g??w-!?l??!?	h"lx??R	!       Z$	?f#U?`???י??g??w-!?l??!?	h"lx??JCPU_ONLYYn??f3@b Y      Y@q?g8??>U@"?
both?Your program is POTENTIALLY input-bound because 55.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?84.9797% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 