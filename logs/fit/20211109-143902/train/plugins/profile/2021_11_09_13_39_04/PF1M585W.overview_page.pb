?($	??7???????u??????(\????!?B?i?q??$	2X??O?@O?e*һ@??QMU?@!?????3@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??b?=??F????x??A?/L?
F??Ym???{???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?1w-!????a??4??AX9??v???YM?O???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&|a2U0*??F%u???A"?uq??Y???<,Ԛ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?Ǻ????
h"lxz??A??y?)??YHP?s??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???Q?????(???A?n?????YV}??b??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&? ?	???x??#????Am???????Y??ׁsF??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&v?????????镲??Aŏ1w-??Yݵ?|г??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&gDio?????St$????A??	h"l??Y?
F%u??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&7?A`????w-!?l??Aё\?C???Ya2U0*???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	?):?????,C????Ax??#????Y?St$????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
?St$?????C??????A?e??a???Y ?o_Ι?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?-????????????A9??v????Yg??j+???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&q???h ??KY?8????A}гY????Y???Mb??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??H?}??&S??:??An????Y	?c???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????????V?/???A??y?):??Y??W?2ġ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??d?`T???S㥛???Ap_?Q??Y?(??0??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???&S?? o?ŏ??AZd;?O???Y?~j?t???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?B?i?q???k	??g??Aq=
ףp??Y?b?=y??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Gr??????^)????A}гY????Y?W[?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???T????,Ԛ????Aq???h ??Y9??v????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??(\????Q?|a2??AF%u???Y{?G?z??*	?????A?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?:pΈ??!?????;A@)J+???1??4??@:Preprocessing2F
Iterator::Model?$??C??!??:*??A@)?S㥛???1Q?K6?7@:Preprocessing2U
Iterator::Model::ParallelMapV2??Q????!"T<Җ)@)??Q????1"T<Җ)@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip7?[ A??!(??j6P@)B`??"۹?1??㝹
(@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate?!??u???!?K????*@)%u???1?z?2K?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice???S㥫?!?HJ?@)???S㥫?1?HJ?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??&S??!n???1@)??d?`T??1?B?&@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*M??St$??!hq??Ȅ@)M??St$??1hq??Ȅ@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 51.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9ޘ??wP@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	_g?,4???J8?????Q?|a2??!?k	??g??	!       "	!       *	!       2$	??J???2???uİ?F%u???!}гY????:	!       B	!       J$	??q???????$??a2U0*???!m???{???R	!       Z$	??q???????$??a2U0*???!m???{???JCPU_ONLYYޘ??wP@b Y      Y@qUw(??fD@"?
both?Your program is POTENTIALLY input-bound because 51.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?40.8009% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 