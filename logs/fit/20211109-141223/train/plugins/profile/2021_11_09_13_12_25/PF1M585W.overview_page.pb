?($	?h?t?????0??S???5^?I??!?j+?????$	???}?@?EN???@??j?\g@!?㱓X/@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?0?*???r?鷯??AV-????Y?:pΈҮ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?A?f???o?ŏ1??AI.?!????Yn????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?X?? ??!?lV}??A?Zd;??Y??d?`T??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?0?*????Q???A?m4??@??Yj?t???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails& A?c?]?????_vO??A?	h"lx??Ye?X???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&5^?I???H?}8??A+??ݓ???Y??ׁsF??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?,C??????	h"??A?W?2??Y??ZӼ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?t?V???ͪ??V??A
ףp=
??YM?O???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&V-?????/n????A?\m?????Y?J?4??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	?b?=y???p=
ף??Ax$(~???Y?J?4??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
?j+???????^)??An4??@???Y?/?'??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&KY?8??????%䃞??A??????Y???<,Ԛ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??y?):???-???1??A?ŏ1w??YDio??ɤ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&&S??:??I??&??A`vOj??Y0*??D??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??{??P???A`??"??AV-?????Y?a??4???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?T???N??u????A??|?5^??Y?
F%u??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&O@a???????H??A????Y/n????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?Q?|???H?}??Ac?ZB>???Y???_vO??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??? ?r????m4????A?]K?=??Y%u???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??ܵ???Tt$?????A???h o??Y??ǘ????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?5^?I??=
ףp=??A????߾??Y?j+??ݓ?*	     ??@2Z
#Iterator::Model::ParallelMapV2::Zip1?Zd??!?LTKQ@)!?rh????1$ͱ??>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatS??:??!???&??8@)(??y??1????w?6@:Preprocessing2F
Iterator::Model?|?5^???!,?ϮҶ?@)ۊ?e????1??8i??2@:Preprocessing2U
Iterator::Model::ParallelMapV2 o?ŏ??!?.??)@) o?ŏ??1?.??)@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSliceh??s???!?? 7?@)h??s???1?? 7?@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate??"??~??!Av?5V#@)c?ZB>???1??e?4?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapݵ?|г??!&????)@)?5?;Nѡ?1?'9??
@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*??ׁsF??!b?M????)??ׁsF??1b?M????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 50.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9x1?h@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	??=(??u{8?h???r?鷯??!??^)??	!       "	!       *	!       2$	?o?t:O????j????????߾??!x$(~???:	!       B	!       J$	ڠ? W???F?vA???%u???!?J?4??R	!       Z$	ڠ? W???F?vA???%u???!?J?4??JCPU_ONLYYx1?h@b Y      Y@q|o~?C@"?
both?Your program is POTENTIALLY input-bound because 50.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?38.063% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 