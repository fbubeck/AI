?&$	f??E?	????Ŀ??????4?q??!)??R"??$	?P(kw?@???7????O?x\?@!?'???@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0q?::?F??9Q?????A><K?P??Y?6??????rtrain 81"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?l??p??7T??7???A?f????Y??ؖg??rtrain 82"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0i??????Q}>??A?Ǚ&l???Y??5??W??rtrain 83"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??ȭI???Ii6??`??Az?????Y?$?j???rtrain 84"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?_YiR?????q?@??AH?9????Y??Os?"??rtrain 85"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0T㥛Ġ???:?vٯ??A?Ŧ?B??Y!sePmp??rtrain 86"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails037߈????.9?????A??Tƿ??Y???3????rtrain 87"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0	ƥ*mq????kC?8??A5A?} R??Y?(???ǒ?rtrain 88"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0
?<?????mU??APoF?W???Y?u??Xߐ?rtrain 89"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0d??u???qs* ??A$??????Y?Q?=??rtrain 90"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???4?q??.S??i??A?|?R????Y??oB!??rtrain 91"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0k??*???:?6U????A Q0??Y??;?_???rtrain 92"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?ۂ?????	pz????AuYLl>.??Y&??s|???rtrain 93"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0)??R"??[{??B???A??'?8??Y?ᔹ?F??rtrain 94"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?ŏ1w???p?{????A???)??Y?H/j????rtrain 95"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??ۂ?:???!??????Aе/????Y????ڛ?rtrain 96"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0:Yj??h?????????A??H??_??Yqs* ??rtrain 97*	???Mb??@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat]4d<J%??!?{T??[D@)5|?ƻ??1?k/Z?8A@:Preprocessing2E
Iterator::Root@?#H????!????xSA@)1E?4~???1?=_k?F2@:Preprocessing2T
Iterator::Root::ParallelMapV2Pō[????!]???_0@)Pō[????1]???_0@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlicecB?%U۱?!??%Y?j @)cB?%U۱?1??%Y?j @:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip??4?8???!,)??CVP@)?????~??1?7?̄ @:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*@?ŊL??!??(?j@)@?ŊL??1??(?j@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenateo?????!?]8??)@)\Y???"??1?f%V??@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMapSh?
??!???U?0@)*;??.R??1	C?N@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 43.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9??Eĳa@IL??a?X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	%???????5???.S??i??!	pz????	!       "	!       *	!       2$	ʊ??ڨ??i)?[???|?R????!??'?8??:	!       B	!       J$	??4??@??L߿?b?????3????!?ᔹ?F??R	!       Z$	??4??@??L߿?b?????3????!?ᔹ?F??b	!       JCPU_ONLYY??Eĳa@b qL??a?X@Y      Y@q>?>t9P@"?	
both?Your program is POTENTIALLY input-bound because 43.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?64.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 