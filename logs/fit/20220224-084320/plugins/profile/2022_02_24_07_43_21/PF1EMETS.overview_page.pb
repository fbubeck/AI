?&$	`??a????8????FN????!???U?;??$	Z?????	@vx???T????@?\?@!9{?|ظ@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?nJy???C?y?'??AD??]L??Y{m??]??rtrain 81"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??????????????A?:????Y?=@??̖?rtrain 82"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0sHj?d????L1AG??A??`obH??Y5&?\R??rtrain 83"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?ۄ{e????????}??A???????Y??4?8E??rtrain 84"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?N??:????? ?> ??A[Υ?????Y?L?????rtrain 85"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?! 8????q?{??c??AS]????Y?EE?N???rtrain 86"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?FN????%??r???A?Ϲ?????Y???????rtrain 87"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0	???U?;??H?9??*??A?'??????Y?X??C??rtrain 88"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0
??KqU????x?n?|??A?oB!??Y??}?<??rtrain 89"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?Ŧ?B??}?????A??.\s??Y@?ŊL??rtrain 90"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?V?f????V&?R???A??/-????YW'g(?x??rtrain 91"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0؟??N0?????Y???Az???X??Y\qqTn???rtrain 92"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???Y.??4??`??A??tB???Y??S?K??rtrain 93"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0=dʇ?j??d?? w??A??q?d???Y???%VF??rtrain 94"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0{??B1??uv28J^??AsG?˵h??Y?8h???rtrain 95"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??Z?f?????????Ar?CQ?O??Y????hq??rtrain 96"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0????y???U4??????A8ӅX???Yx???Ĭ??rtrain 97*#??~j ?@)      @=2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat??66;??!?????rB@)#??ݯ??1??NC?6A@:Preprocessing2E
Iterator::Root??}?<??!-??!???@)I???p???1#?@???0@:Preprocessing2T
Iterator::Root::ParallelMapV2?#EdX??!R?`??-@)?#EdX??1R?`??-@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice?"?~???!?P???P&@)?"?~???1?P???P&@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip????}???!uW??Q@)?S??ѵ?1A??&@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap>???6??!?yP?t4@)?#??S ??1N??jT=@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate?V?9?m??!㲈k??.@)?g?u????1B?F!??@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*O??e???!??rU??@)O??e???1??rU??@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 48.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?q?Z??	@It,*ݲ2X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	oZB????????[F??4??`??!?? ?> ??	!       "	!       *	!       2$	??ܤ???l]F\???Ϲ?????!?'??????:	!       B	!       J$	?wF[?D?????Nr\d???}?<??!???????R	!       Z$	?wF[?D?????Nr\d???}?<??!???????b	!       JCPU_ONLYY?q?Z??	@b qt,*ݲ2X@Y      Y@qR=Pv??R@"?	
both?Your program is POTENTIALLY input-bound because 48.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?76.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 