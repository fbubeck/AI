?($		pP?????p??? ??NbX9???!(~??k	??$	??1??C@s?#[??@??l$ܒ@!)?64@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??u????? ?rh???A|??Pk???Y9??m4???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Zd;?O???@a??+??A	??g????YT㥛? ??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?|гY???.???1???A?[ A???Y?? ?rh??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&(~??k	???ׁsF???A?/?'??Yŏ1w-!??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??????6?>W[???AW?/?'??Y?:pΈҞ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&h??|?5??z?,C???A???N@??Y???Q???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?-?????):????A??d?`T??Y?:pΈҞ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??K7?A????y?)??A?7??d???Y	?c???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&V-????Ǻ?????A??ݓ????Y??e?c]??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	??(????2U0*???A?N@a???Yp_?Q??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
L?
F%u?? o?ŏ??AԚ?????YjM????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??K7?A??^K?=???A?ڊ?e???Y?sF????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?x?&1??鷯???A??C?l??Y?<,Ԛ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&/?$????V-??A????߾??Y???{????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&1?Zd??:#J{?/??A???H.??YT㥛? ??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?s??????=yX?5??A㥛? ???Y?p=
ף??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&K?=?U???MbX9??A$???~???Y2??%䃞?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&V}??b??6?>W[???A??x?&1??YvOjM??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??y?):???X?? ??A????z??Y? ?	???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&6<?R?!????&???Aa??+e??Y?a??4???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&NbX9??????镲??A????Y?U???؟?*	?????ʐ@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???_vO??!?tG?7@@)o???T???19\->@:Preprocessing2F
Iterator::Model?{??Pk??!?M?r?4C@)j?t???1??:S??5@:Preprocessing2U
Iterator::Model::ParallelMapV2?:pΈ???!_???6?0@)?:pΈ???1_???6?0@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???H.??!
?
?m?N@)K?=?U??1?Vp|H?&@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice?n??ʱ?!>?? ?@)?n??ʱ?1>?? ?@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate|a2U0*??!?O7j??(@)aTR'????16?vJ>@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap8gDio??!.O?MZ?1@)?ZӼ???1D??b
$@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*??+e???!K????@)??+e???1K????@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 55.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9L?zBߛ@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	̔?'???n?c?'?????镲??!?ׁsF???	!       "	!       *	!       2$	?AQr?@???Y?u???????!??x?&1??:	!       B	!       J$	bܹ??????ܿ?N??p_?Q??!9??m4???R	!       Z$	bܹ??????ܿ?N??p_?Q??!9??m4???JCPU_ONLYYL?zBߛ@b Y      Y@qtD?[?8B@"?
both?Your program is POTENTIALLY input-bound because 55.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?36.4418% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 