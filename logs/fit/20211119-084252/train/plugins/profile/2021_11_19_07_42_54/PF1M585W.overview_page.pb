?($	k??[???????A??ǘ???!????????$	Eou5?@"Q?S\?
@?r?v?@!Xw?o??1@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?$??C?????H.??Ah"lxz???Yc?=yX??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&|??Pk????? ???A??镲??Y}гY????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????????6<?R???A???JY???Y+??Χ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??	h"l???g??s???A?Q???YX?5?;N??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?ʡE?????lV}???A??_vO??YD?l?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??HP??????o??A?\?C????Y????<,??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails& A?c?]??|??Pk???A?C?l????Y??d?`T??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&O@a????8gDio??A??y???Y
ףp=
??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?\?C????ё\?C???A??ܵ???Y䃞ͪϕ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	t???????]K?=??A??{??P??YˡE?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
?????????????A???H??Y0L?
F%??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???????Z??ڊ???A?@??ǘ??YǺ?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&h??s????~?:p???AK?46??Y?1w-!??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&O@a???????V?/??A???h o??Y0*??D??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&9EGr???z?,C???Ay?&1???Y??ׁsF??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&o??ʡ????Q????A??W?2???Y??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?n?????3ı.n???A/?$???Y?HP???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&гY????????????A??&???Yr??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&0L?
F%??;M?O??AH?z?G??Y?]K?=??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?G?z??2U0*???AR'??????Y"??u????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&A??ǘ???u????A?W?2??Y
ףp=
??*	43333g?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?O??n??!?????+A@)?/?'??1?1 ?Q?@:Preprocessing2F
Iterator::ModelW[??????!!RlӑRB@)? ?	???1??>u?x7@:Preprocessing2U
Iterator::Model::ParallelMapV2ݵ?|г??!'[3cY*@)ݵ?|г??1'[3cY*@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip*:??H??!ޭ?,n?O@)?c?ZB??1?T:T??&@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice?߾?3??!??P?C@)?߾?3??1??P?C@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenateݵ?|г??!'[3cY*@)?J?4??1b$:ٚ@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap$???????!?)#?y?1@)?-????1??%??S@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*???~?:??!???/(@)???~?:??1???/(@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 51.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9[?x ٳ@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	O<?Lu??Յ??xK??u????!Z??ڊ???	!       "	!       *	!       2$	t$???????"b?????W?2??!???JY???:	!       B	!       J$	?HGg??????O?:????d?`T??!c?=yX??R	!       Z$	?HGg??????O?:????d?`T??!c?=yX??JCPU_ONLYY[?x ٳ@b Y      Y@q>9\8>@"?
both?Your program is POTENTIALLY input-bound because 51.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?30.2188% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 