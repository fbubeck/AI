?($	?c?C.??????????u????!??JY?8??$	???*?Q@???Y? @{?)	? @!??#??(@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$u????1?Zd??A?C?l????Y??JY?8??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&h??|?5??}гY????A8gDio??Yn????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&*:??H????m4????A	?^)???YL7?A`???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??ZӼ???????????A5^?I??Y? ?	???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&)\???(????a??4??A?Zd;??Y"??u????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&r?鷯????C?l??A???1????Ye?X???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?%䃞??????JY???AZd;?O???Y?b?=y??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&[B>?٬?????????A???????Yvq?-??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Gx$(???f??j+??A~??k	???Y46<???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	'1?Z?????镲??A?^)???Y%u???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
+??ݓ????<,Ԛ???A??n????YU???N@??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??JY?8????~j?t??A*:??H??Y?,C????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??ݓ????(~??k	??A
h"lxz??Y??ׁsF??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&mV}??b???? ?	??A???ׁs??Y?4?8EG??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&&䃞ͪ???'????A???߾??Y?
F%u??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??s????s??A???A?:M???Y	?^)ˠ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???Mb????a??4??A?A?f????YT㥛? ??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails& c?ZB>??????_v??A?z?G???Y???x?&??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&$(~??k??h??|?5??A????Y??+e???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&t??????\ A?c???A???Mb??Y?sF????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??ʡE????[ A???A????o??YT㥛? ??*	fffff??@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatZd;?O???!L?(?_B@)?߾?3??1]G?A@:Preprocessing2F
Iterator::Model㥛? ???!?o(?A@)??T?????1????G5@:Preprocessing2U
Iterator::Model::ParallelMapV2?ŏ1w??!>F?
?+@)?ŏ1w??1>F?
?+@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zips??A??!r?k?7P@)Ӽ?ɵ?1v??5Uy$@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice?;Nё\??!?0t??@)?;Nё\??1?0t??@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate6?;Nё??!x???y?*@)?^)?Ǫ?1? ??W*@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapF%u???!qxA?<?1@)??\m????1?*???@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*^K?=???!?~_;8N@)^K?=???1?~_;8N@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 48.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?=~??t@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	gS?`???DaA????1?Zd??!(~??k	??	!       "	!       *	!       2$	Q?!???ΞU??????C?l????!*:??H??:	!       B	!       J$	IK??ھ??@?????%u???!?,C????R	!       Z$	IK??ھ??@?????%u???!?,C????JCPU_ONLYY?=~??t@b Y      Y@q¸Lr"A@"?
both?Your program is POTENTIALLY input-bound because 48.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?34.0948% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 