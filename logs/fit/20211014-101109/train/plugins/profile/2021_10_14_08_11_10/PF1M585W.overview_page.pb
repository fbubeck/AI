?($	???<k????U?????Pk?w???!x$(~??$	???g@IMݘ?@@?:S;??!??6Y?0@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$2??%?????镲q??As??A??Yh??|?5??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&x$(~????0?*??A??{??P??Y?U???؟?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????????ݓ??Z??A???????Y?J?4??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?a??4????Q?????A?t?V??Y ?o_Ι?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&r??????`??"????A-C??6??Y?	h"lx??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??y?)???e??a???AbX9????Y?????K??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?/?$?????????A?w??#???Y?~j?t???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?٬?\m??S??:??A??:M??Y??@??ǘ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Z??ڊ?????<,Ԛ??A?O??n??Y;?O??n??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	P??n???6<?R???A????9#??YDio??ɔ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
????????!?lV}??A.???1???YL7?A`???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&@a??+??M??St$??Aݵ?|г??Y?o_???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??????|??Pk???A??{??P??Y?St$????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??HP????JY?8??A?[ A?c??Y/n????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??"??~???e??a???A?p=
ף??Y??ǘ????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&8gDio??%u???A?S㥛???Y	?^)ː?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?X?? ??F????x??AF????x??Y?St$????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??%䃞??
h"lxz??A????_v??Y??ǘ????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????????f??j+??A?A?f????Y??ܵ?|??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?\m?????\ A?c???A?Zd;??Y??ׁsF??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?Pk?w???h??|?5??AǺ????Y0*??D??*	?????Ո@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??%䃞??!]??)-RA@)?ͪ??V??1Aw??
@@:Preprocessing2F
Iterator::Model?u?????!?dp}?XA@)X9??v??1?Bt6@:Preprocessing2U
Iterator::Model::ParallelMapV2?'????!zb?q?7)@)?'????1zb?q?7)@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??S㥛??!??G??SP@)Ǻ?????1h?ͤ?&@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap$(~????!???Jd3@)e?`TR'??1r??h??@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?@??ǘ??!H?G6!@)?@??ǘ??1H?G6!@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?????K??!e??`??&@)??\m????1?5izx?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*{?G?z??!???n%"@){?G?z??1???n%"@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 47.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?Ĩh??@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	P-g9???uBm?i??h??|?5??!??0?*??	!       "	!       *	!       2$	??ެm?????A?]??Ǻ????!bX9????:	!       B	!       J$	???{Kz???v5Z?????ܵ?|??!h??|?5??R	!       Z$	???{Kz???v5Z?????ܵ?|??!h??|?5??JCPU_ONLYY?Ĩh??@b Y      Y@q???. mU@"?
both?Your program is POTENTIALLY input-bound because 47.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?85.7051% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 